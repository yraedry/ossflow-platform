"""Servicio de dubbing: proxy fino hacia el backend dubbing-generator.

Responsabilidades:

* Traducir paths host → container con ``library_path`` y
  ``ossflow_api.shared.paths.to_container_path``.
* Resolver el ``voice_profile`` desde el sidecar ``.bjj-meta.json`` cuando
  el body no lo trae (vía ``ossflow_api.shared.voice_profiles``).
* Hacer ``POST/GET/PUT {base_url}{path}`` con timeouts y propagar
  errores HTTP del backend al caller.
* Servir los sidecars ``*.dub-qa.json`` desde el filesystem (los
  endpoints ``/qa`` y ``/qa/instructional/{name}`` no salen al
  microservicio, leen disco directamente).

Mantiene el comportamiento exacto de ``api/dubbing.py`` original; los
cambios son sólo de empaquetado para encajar en el patrón vertical
slice y para que las dependencias sean inyectables en tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from ossflow_api.shared.exceptions import ApiError, NotFoundError, UpstreamError
from ossflow_api.shared.paths import to_container_path

from .schemas import AnalyzeBody, VoiceTranscriptBody

log = logging.getLogger(__name__)


class _BadPath(ApiError):
    """``to_container_path`` rechazó el path host."""

    status_code = 400


class DubbingService:
    """Cliente de proxy hacia dubbing-generator.

    ``library_path``, ``dubbing_url`` y los helpers se inyectan por
    constructor para que los tests puedan reemplazarlos sin tocar la
    BD de settings ni el filesystem global.
    """

    def __init__(
        self,
        *,
        library_path: Optional[str],
        dubbing_url: str,
        voice_profile_loader: Callable[[str], str],
        scan_cache_loader: Callable[[], Optional[dict]],
    ) -> None:
        self._library_path = library_path
        self._dubbing_url = dubbing_url.rstrip("/")
        self._load_voice_profile = voice_profile_loader
        self._load_scan_cache = scan_cache_loader

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _translate_path(self, host_path: str) -> str:
        """Convierte un path host a path container o lo deja igual.

        Si no hay ``library_path`` configurado se devuelve sin tocar para
        preservar compatibilidad con tests/desarrollo local.
        """
        if not self._library_path:
            return host_path
        try:
            return to_container_path(host_path, self._library_path)
        except ValueError as exc:
            raise _BadPath(str(exc)) from exc

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        url = f"{self._dubbing_url}{path}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.post(url, json=payload)
            except httpx.HTTPError as exc:
                raise UpstreamError(f"dubbing-generator: {exc}") from exc
        if r.status_code >= 400:
            raise UpstreamError(r.text, status_code=r.status_code)
        return r.json()

    async def _get(self, path: str, *, timeout: float = 5.0) -> dict[str, Any]:
        url = f"{self._dubbing_url}{path}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.get(url)
            except httpx.HTTPError as exc:
                raise UpstreamError(f"dubbing-generator: {exc}") from exc
        if r.status_code >= 400:
            raise UpstreamError(r.text, status_code=r.status_code)
        return r.json()

    async def _put(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        url = f"{self._dubbing_url}{path}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.put(url, json=payload)
            except httpx.HTTPError as exc:
                raise UpstreamError(f"dubbing-generator: {exc}") from exc
        if r.status_code >= 400:
            raise UpstreamError(r.text, status_code=r.status_code)
        return r.json()

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def list_voices(self) -> dict[str, Any]:
        """Lista los WAV de referencia (voces ES) disponibles en el backend."""
        return await self._get("/voices", timeout=5.0)

    async def save_voice_transcript(
        self, filename: str, body: VoiceTranscriptBody
    ) -> dict[str, Any]:
        """Persiste una transcripción de referencia como sidecar junto al WAV."""
        return await self._put(
            f"/voices/{filename}/transcript",
            {"transcript": body.transcript},
            timeout=5.0,
        )

    def get_dub_qa(self, video_path: str) -> dict[str, Any]:
        """Devuelve el sidecar ``{stem}.dub-qa.json`` generado por el dubbing pipeline.

        404 si no se ha generado QA todavía — el frontend lo trata como
        "badge oculto" en vez de error.
        """
        vp = Path(video_path)
        sidecar = vp.with_suffix("").with_name(f"{vp.stem}.dub-qa.json")
        if not sidecar.exists():
            raise NotFoundError("No QA sidecar for this video")
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ApiError(f"Invalid QA sidecar: {exc}", status_code=500) from exc

    def get_instructional_qa(self, name: str) -> dict[str, Any]:
        """QA agregado para todos los capítulos doblados de un instruccional.

        Lee cada sidecar ``{stem}.dub-qa.json`` y devuelve una lista plana
        con la misma forma que el endpoint por-vídeo, más un bloque
        ``summary`` (medias, peor capítulo) para que el frontend pinte
        cabeceras sin recalcular.

        Capítulos sin sidecar aparecen con ``qa: null``; la UI los
        muestra como "sin QA" para indicar al usuario qué hace falta
        re-doblar.
        """
        data = self._load_scan_cache()
        if data is None:
            raise NotFoundError("no scan cache")

        items = data.get("instructionals", []) if isinstance(data, dict) else []
        match = next((it for it in items if it.get("name") == name), None)
        if match is None:
            raise NotFoundError("instructional not found")

        chapters: list[dict[str, Any]] = []
        mos_values: list[float] = []
        levels = {"green": 0, "amber": 0, "red": 0}
        worst: dict[str, Any] | None = None
        worst_score = float("inf")

        for v in (match.get("videos") or []):
            if not isinstance(v, dict):
                continue
            vp_str = v.get("path") or ""
            if not vp_str:
                continue
            vp = Path(vp_str)
            sidecar = vp.with_name(f"{vp.stem}.dub-qa.json")
            entry: dict[str, Any] = {
                "filename": v.get("filename") or vp.name,
                "path": vp_str,
                "has_dubbing": bool(v.get("has_dubbing") or v.get("has_dubbed")),
                "qa": None,
            }
            if sidecar.exists():
                try:
                    entry["qa"] = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    entry["qa"] = {"error": "invalid sidecar"}

            qa = entry["qa"] if isinstance(entry.get("qa"), dict) else None
            verdict = (qa or {}).get("verdict") or {}
            lvl = verdict.get("level")
            if lvl in levels:
                levels[lvl] += 1
            mos = verdict.get("mos")
            if isinstance(mos, (int, float)):
                mos_values.append(float(mos))
                if mos < worst_score:
                    worst_score = float(mos)
                    worst = {"filename": entry["filename"], "mos": mos}

            chapters.append(entry)

        avg_mos = round(sum(mos_values) / len(mos_values), 2) if mos_values else None
        # ``with_qa`` = sidecar válido (con verdict). Los corruptos cuentan
        # como missing; de otro modo la UI mostraría un contador optimista
        # pero sin datos.
        with_qa = sum(
            1
            for c in chapters
            if isinstance(c.get("qa"), dict)
            and isinstance(c["qa"].get("verdict"), dict)
        )
        summary = {
            "total_chapters": len(chapters),
            "with_qa": with_qa,
            "levels": levels,
            "avg_mos": avg_mos,
            "worst": worst,
        }

        return {"name": name, "summary": summary, "chapters": chapters}

    async def restart(self) -> dict[str, Any]:
        """Proxy del restart graceful (libera VRAM; Docker reinicia el contenedor).

        El contenedor mata su propio PID 1 ~0.5 s después de responder.
        Usamos timeout corto para tolerar que la conexión caiga
        mid-response sin bubbleear un 502 al usuario.
        """
        try:
            return await self._post("/maintenance/restart", {}, timeout=5.0)
        except UpstreamError:
            # El contenedor cayó antes de responder — ese ES el camino feliz.
            return {"ok": True, "message": "Reiniciando dubbing-generator…"}

    async def analyze(self, body: AnalyzeBody) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "video_path": self._translate_path(body.video_path),
            "synthesize": body.synthesize,
        }
        if body.srt_path:
            payload["srt_path"] = self._translate_path(body.srt_path)
        if body.max_phrases is not None:
            payload["max_phrases"] = body.max_phrases
        vp = body.voice_profile or self._load_voice_profile(body.video_path)
        if vp:
            payload["voice_profile"] = vp
        # Synthesis ejecuta TTS sobre N frases → muy lento; timeout generoso.
        timeout = 1200.0 if body.synthesize else 60.0
        return await self._post("/analyze", payload, timeout=timeout)
