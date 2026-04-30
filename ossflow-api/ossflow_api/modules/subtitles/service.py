"""Servicio de subtitles: proxy fino hacia el backend subtitle-generator.

Responsabilidades:

* Traducir paths host → container usando ``library_path`` y
  ``ossflow_api.shared.paths.to_container_path``.
* Resolver provider/model/api_key de translate desde settings cuando el
  body no los trae, con fallback opcional.
* Hacer ``POST {base_url}{path}`` con timeouts por endpoint y propagar
  errores HTTP del backend al caller.

Mantiene el comportamiento exacto de ``api/subtitles.py`` original; los
cambios son sólo de empaquetado para encajar en el patrón vertical
slice y para que las dependencias sean inyectables en tests.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import httpx

from ossflow_api.shared.exceptions import ApiError, UpstreamError, ValidationError
from ossflow_api.shared.paths import to_container_path

from .schemas import AnalyzeBody, ApplyBody, RegenerateBody, TranslateBody, ValidateBody

log = logging.getLogger(__name__)


class _BadPath(ApiError):
    """``to_container_path`` rechazó el path host."""

    status_code = 400


class SubtitlesService:
    """Cliente de proxy hacia subtitle-generator.

    ``library_path`` y ``setting_getter`` se inyectan por constructor para
    que los tests puedan reemplazarlos sin tocar la BD de settings.
    """

    def __init__(
        self,
        *,
        library_path: Optional[str],
        subs_url: str,
        setting_getter: Callable[[str], Any],
    ) -> None:
        self._library_path = library_path
        self._subs_url = subs_url.rstrip("/")
        self._get_setting = setting_getter

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
        url = f"{self._subs_url}{path}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.post(url, json=payload)
            except httpx.HTTPError as exc:
                raise UpstreamError(f"subtitle-generator: {exc}") from exc
        if r.status_code >= 400:
            # Propaga el status original del backend (idéntico al
            # comportamiento del proxy legacy).
            raise UpstreamError(r.text, status_code=r.status_code)
        return r.json()

    def _key_for_provider(
        self, provider: str, override: Optional[str]
    ) -> Optional[str]:
        if override:
            return override
        p = (provider or "").lower().strip()
        if p == "openai":
            return self._get_setting("openai_api_key")
        if p == "ollama":
            return None  # ollama no necesita api_key
        return None

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def validate(self, body: ValidateBody) -> dict[str, Any]:
        payload = {"srt_path": self._translate_path(body.srt_path)}
        return await self._post("/validate", payload, timeout=30.0)

    async def regenerate_segment(self, body: RegenerateBody) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "srt_path": self._translate_path(body.srt_path),
            "segment_idx": body.segment_idx,
            "context_seconds": body.context_seconds,
            "model": body.model,
            "language": body.language,
        }
        if body.video_path:
            payload["video_path"] = self._translate_path(body.video_path)
        return await self._post("/regenerate-segment", payload, timeout=300.0)

    async def apply_segment(self, body: ApplyBody) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "srt_path": self._translate_path(body.srt_path),
            "segment_idx": body.segment_idx,
            "text": body.text,
        }
        if body.start is not None:
            payload["start"] = body.start
        if body.end is not None:
            payload["end"] = body.end
        return await self._post("/apply-segment", payload, timeout=30.0)

    async def clear_locks(self) -> dict[str, Any]:
        """Limpia locks colgados de HuggingFace en subtitle-generator."""
        return await self._post(
            "/maintenance/clear-hf-locks", {}, timeout=15.0
        )

    async def restart(self) -> dict[str, Any]:
        """Solicita un restart graceful al backend (libera VRAM)."""
        return await self._post("/maintenance/restart", {}, timeout=15.0)

    async def translate(self, body: TranslateBody) -> dict[str, Any]:
        provider = (
            body.provider
            or self._get_setting("translation_provider")
            or "ollama"
        ).lower()
        fallback = (
            body.fallback_provider
            or self._get_setting("translation_fallback_provider")
            or ""
        ).lower() or None
        model = body.model or self._get_setting("translation_model")

        payload: dict[str, Any] = {
            "srt_path": self._translate_path(body.srt_path),
            "target_lang": body.target_lang,
            "source_lang": body.source_lang,
            "provider": provider,
        }
        if model:
            payload["model"] = model
        if body.formality:
            payload["formality"] = body.formality

        api_key = self._key_for_provider(provider, body.api_key)
        if provider != "ollama" and not api_key:
            raise ValidationError(
                f"{provider} API key missing (set it in Settings)",
                status_code=400,
            )
        if api_key:
            payload["api_key"] = api_key

        if fallback and fallback != provider:
            fb_key = self._key_for_provider(fallback, body.fallback_api_key)
            if fallback == "ollama" or fb_key:
                payload["fallback_provider"] = fallback
                if fb_key:
                    payload["fallback_api_key"] = fb_key

        if body.out_path:
            payload["out_path"] = self._translate_path(body.out_path)
        if body.dubbing_mode:
            payload["dubbing_mode"] = True
            cps = body.dubbing_cps or self._get_setting("translation_dubbing_cps")
            if cps:
                payload["dubbing_cps"] = float(cps)
        return await self._post("/translate", payload, timeout=600.0)

    async def analyze(self, body: AnalyzeBody) -> dict[str, Any]:
        payload = {
            "video_path": self._translate_path(body.video_path),
            "language": body.language,
            "model": body.model,
        }
        return await self._post("/analyze", payload, timeout=600.0)
