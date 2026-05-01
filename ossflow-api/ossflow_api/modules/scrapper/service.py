"""Servicio del módulo scrapper (proxy oracle).

Bridge entre el ``processor-api`` y el subsistema ``oracle`` del
microservicio ``chapter-splitter`` (comunicación HTTP únicamente, sin
import de ``chapter_splitter``).

Responsabilidades preservadas literalmente del antiguo
``api/oracle.py``:

* Resolución y validación de paths bajo ``library_path`` (anti-traversal).
* Lectura/escritura atómica del sidecar ``.bjj-meta.json``.
* Validación mínima del shape ``OracleResult``.
* Auto-descarga de poster (con strip de mutaciones de Shopify y trim de
  bordes negros).
* Heurísticas para derivar título/autor desde el nombre de la carpeta
  cuando el sidecar no los trae.
* Proxy HTTP hacia los endpoints ``/oracle/*`` del backend.

Cambios estructurales:

* ``scan_cache`` ahora se obtiene por DI (``scan_cache_loader``) en
  lugar de instanciar uno propio. Cierra acoplamiento #8 — la versión
  legacy creaba un ``ScanCache`` paralelo al de ``app.py`` que
  divergía en runtime cuando ambos escribían al mismo fichero.
* ``library_path_loader`` y ``patch_poster`` también se inyectan para
  que los tests puedan reemplazarlos sin tocar ``api.settings`` ni
  ``api.scan_cache`` en tiempo de import.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import unquote

import httpx

from ossflow_api.shared.exceptions import ApiError, NotFoundError, UpstreamError

log = logging.getLogger(__name__)


SIDECAR_NAME = ".bjj-meta.json"
DEFAULT_TIMEOUT = 30.0


class _BadPath(ApiError):
    """``_resolve_instructional`` rechazó el path host."""

    status_code = 400


class _PathOutsideLibrary(ApiError):
    status_code = 403


class _InvalidOracle(ApiError):
    status_code = 422


# ---------------------------------------------------------------------------
# Heurísticas y helpers puros (no necesitan estado del servicio)
# ---------------------------------------------------------------------------

_POSTER_STEMS = ("poster", "cover", "folder")
_POSTER_EXTS = ("jpg", "jpeg", "png", "webp")


def _has_local_poster(folder: Path) -> bool:
    for stem in _POSTER_STEMS:
        for ext in _POSTER_EXTS:
            if (folder / f"{stem}.{ext}").exists():
                return True
            if (folder / f"{stem}.{ext.upper()}").exists():
                return True
    return False


def _trim_black_borders(
    path: Path, *, threshold: int = 18, min_keep_ratio: float = 0.4
) -> bool:
    """Recorta bordes negros sólidos in-place de un poster.

    BJJFanatics sirve muchos posters como JPEG portrait con el arte real
    centrado entre bandas negras. Otros productos vienen sin borde y no
    necesitan trim. Detectamos por umbralizado a máscara binaria + bbox.

    Guards:
      * ``threshold`` — píxeles más oscuros que esto en todos los canales
        cuentan como borde. 18 captura el ruido JPEG alrededor del negro
        verdadero sin comerse la ropa oscura del arte.
      * ``min_keep_ratio`` — nunca recortar más del 60% en cualquier
        dimensión. Frena falsos positivos en arte genuinamente oscuro.

    Devuelve True si modificó el fichero.
    """
    try:
        from PIL import Image, ImageChops  # noqa: WPS433 — dep opcional
    except ImportError:
        log.debug("Pillow not available, skipping poster trim")
        return False
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            bg = Image.new("RGB", rgb.size, (0, 0, 0))
            diff = ImageChops.difference(rgb, bg)
            mask = diff.convert("L").point(lambda v: 255 if v > threshold else 0)
            bbox = mask.getbbox()
            if not bbox:
                return False  # imagen completamente negra — la dejamos
            left, top, right, bottom = bbox
            width, height = rgb.size
            new_w = right - left
            new_h = bottom - top
            if new_w >= width and new_h >= height:
                return False  # nada que recortar
            if new_w < width * min_keep_ratio or new_h < height * min_keep_ratio:
                # Sospechoso — recortaríamos >60%. Probablemente arte
                # genuinamente oscuro; respetamos el original.
                log.info(
                    "skipping poster trim for %s (would keep %dx%d of %dx%d)",
                    path.name, new_w, new_h, width, height,
                )
                return False
            cropped = rgb.crop(bbox)
            cropped.save(path, quality=92, optimize=True)
            log.info(
                "trimmed poster %s: %dx%d → %dx%d",
                path.name, width, height, new_w, new_h,
            )
            return True
    except Exception as exc:  # noqa: BLE001
        log.warning("poster trim failed for %s: %s", path, exc)
        return False


def _strip_shopify_thumb(url: str) -> str:
    """Quita mutaciones de tamaño del CDN Shopify (query + path).

    Sidecars antiguos guardaron URLs con ``?crop=center&height=300&width=300``
    o un sufijo ``_NNNxMMM`` en el path. Sirven un thumbnail centrado de
    300×300 que pierde la parte superior del arte. Limpiamos ambas formas
    para que el redownload obtenga el original a máxima resolución.
    Equivalente a ``BjjFanaticsProvider._strip_shopify_size`` del backend.
    """
    if (
        "cdn.shop" not in url
        and "cdn.shopify" not in url
        and "bjjfanatics.com/cdn" not in url
    ):
        return url
    try:
        base, _sep, query = url.partition("?")
        base = re.sub(r"_\d+x\d+(?=\.[A-Za-z0-9]+$)", "", base)
        if not query:
            return base
        keep = []
        for part in query.split("&"):
            key = part.split("=", 1)[0].lower()
            if key in {"width", "height", "crop", "pad_color"}:
                continue
            keep.append(part)
        return base + ("?" + "&".join(keep) if keep else "")
    except Exception:  # noqa: BLE001
        return url


def _validate_oracle_result(data: Any) -> dict[str, Any]:
    """Valida shape mínimo de ``OracleResult``. Lanza ``_InvalidOracle`` si no encaja."""
    if not isinstance(data, dict):
        raise _InvalidOracle("OracleResult must be an object")

    product_url = data.get("product_url", "")
    scraped_at = data.get("scraped_at", "")
    volumes = data.get("volumes", [])

    if not isinstance(product_url, str):
        raise _InvalidOracle("product_url must be string")
    if not isinstance(scraped_at, str):
        raise _InvalidOracle("scraped_at must be string")
    if not isinstance(volumes, list):
        raise _InvalidOracle("volumes must be list")

    clean_volumes: list[dict[str, Any]] = []
    for vi, vol in enumerate(volumes):
        if not isinstance(vol, dict):
            raise _InvalidOracle(f"volumes[{vi}] must be object")
        number = vol.get("number")
        chapters = vol.get("chapters", [])
        total_duration_s = vol.get("total_duration_s", 0)
        if not isinstance(number, int):
            raise _InvalidOracle(f"volumes[{vi}].number must be int")
        if not isinstance(chapters, list):
            raise _InvalidOracle(f"volumes[{vi}].chapters must be list")
        if not isinstance(total_duration_s, (int, float)):
            raise _InvalidOracle(
                f"volumes[{vi}].total_duration_s must be number"
            )

        clean_chapters: list[dict[str, Any]] = []
        for ci, ch in enumerate(chapters):
            if not isinstance(ch, dict):
                raise _InvalidOracle(
                    f"volumes[{vi}].chapters[{ci}] must be object"
                )
            title = ch.get("title", "")
            start_s = ch.get("start_s")
            end_s = ch.get("end_s")
            if not isinstance(title, str):
                raise _InvalidOracle(f"chapters[{ci}].title must be string")
            if not isinstance(start_s, (int, float)) or not isinstance(
                end_s, (int, float)
            ):
                raise _InvalidOracle(
                    f"chapters[{ci}] start_s/end_s must be number"
                )
            clean_chapters.append({
                "title": title,
                "start_s": float(start_s),
                "end_s": float(end_s),
            })

        clean_volumes.append({
            "number": number,
            "chapters": clean_chapters,
            "total_duration_s": float(total_duration_s),
        })

    out: dict[str, Any] = {
        "product_url": product_url,
        "scraped_at": scraped_at,
        "volumes": clean_volumes,
    }
    if "provider_id" in data and isinstance(data["provider_id"], str):
        out["provider_id"] = data["provider_id"]
    if (
        "poster_url" in data
        and isinstance(data["poster_url"], str)
        and data["poster_url"]
    ):
        out["poster_url"] = data["poster_url"]
    return out


def _derive_title_author(folder: Path, meta: dict[str, Any]) -> tuple[str, str]:
    """Deriva (title, author) del meta o, en su defecto, del nombre del folder.

    Patrones vistos:
      * ``"Title - Author"`` (convención de la librería — derecha es persona)
      * ``"Author - Title"`` (legacy)
      * ``"Title by Author"``

    Heurística: en split por ``" - "``, el lado con ≤3 palabras es autor.
    Si ambos lados >3, se prefiere DERECHA como autor (convención de la
    librería).
    """
    title = meta.get("topic") or ""
    author = meta.get("instructor") or ""
    if title and author:
        return title, author

    name = folder.name
    if " - " in name and not author:
        left, right = [s.strip() for s in name.split(" - ", 1)]
        left_words = len(left.split())
        right_words = len(right.split())
        if right_words <= 3 and left_words > right_words:
            title = title or left
            author = author or right
        elif left_words <= 3 and right_words > left_words:
            author = author or left
            title = title or right
        else:
            # Ambiguo — default a "Title - Author" (convención).
            title = title or left
            author = author or right
    elif " by " in name.lower() and not author:
        idx = name.lower().rindex(" by ")
        title = title or name[:idx].strip()
        author = author or name[idx + 4:].strip()
    else:
        title = title or name
    return title, author


def _read_meta(folder: Path) -> dict[str, Any]:
    sidecar = folder / SIDECAR_NAME
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        log.warning("failed to read meta %s: %s", sidecar, exc)
        return {}


def _write_meta_atomic(folder: Path, meta: dict[str, Any]) -> None:
    sidecar = folder / SIDECAR_NAME
    tmp = folder / (SIDECAR_NAME + ".tmp")
    payload = json.dumps(meta, indent=2, ensure_ascii=False)
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, sidecar)


# ---------------------------------------------------------------------------
# Servicio
# ---------------------------------------------------------------------------


# TODO(T23): cuando exista el módulo ``library``, ``scan_cache_loader``
# desaparecerá y este servicio recibirá directamente un repositorio o
# manager de la librería. Hoy seguimos haciendo import diferido en
# ``dependencies.py`` apuntando al ``_scan_cache`` global de ``app.py``.
ScanCacheLoader = Callable[[], Any]
PosterPatcher = Callable[[Any, str, str], None]


class ScrapperService:
    """Proxy hacia el subsistema oracle del chapter-splitter.

    Las dependencias I/O (``library_path``, ``scan_cache``,
    ``patch_poster_in_cache``) se inyectan vía constructor para permitir
    sustituirlas en tests sin tocar globals.
    """

    def __init__(
        self,
        *,
        splitter_url: str,
        library_path_loader: Callable[[], Optional[str]],
        scan_cache_loader: ScanCacheLoader,
        patch_poster: PosterPatcher,
    ) -> None:
        self._splitter_url = splitter_url.rstrip("/")
        self._library_path_loader = library_path_loader
        self._scan_cache_loader = scan_cache_loader
        self._patch_poster = patch_poster

    # ------------------------------------------------------------------
    # Resolución de paths
    # ------------------------------------------------------------------

    def _resolve_instructional(self, raw_path: str) -> Path:
        """Decode + valida que ``raw_path`` esté bajo ``library_path``."""
        decoded = unquote(raw_path).strip()
        if not decoded:
            raise _InvalidOracle("empty instructional path")

        lib = self._library_path_loader()
        if not lib:
            raise _BadPath("library_path no configurado")

        # El frontend suele enviar el path host completo; intentamos como
        # absoluto primero y caemos a relativo bajo la librería.
        candidate = Path(decoded)
        if not candidate.is_absolute():
            candidate = Path(lib) / decoded

        try:
            resolved = candidate.resolve()
            lib_resolved = Path(lib).resolve()
            resolved.relative_to(lib_resolved)
        except (OSError, ValueError):
            raise _PathOutsideLibrary("path outside library")

        if not resolved.exists() or not resolved.is_dir():
            raise NotFoundError("instructional not found")
        return resolved

    # ------------------------------------------------------------------
    # Poster auto-download
    # ------------------------------------------------------------------

    async def _download_poster_if_missing(
        self,
        folder: Path,
        poster_url: Optional[str],
        *,
        force: bool = False,
    ) -> Optional[str]:
        """Descarga ``poster_url`` a ``folder/poster.<ext>``.

        Si ``force`` es True, elimina cualquier poster local existente
        antes. Devuelve el filename guardado o None si se saltó/falló.
        """
        if not poster_url:
            return None
        # Defensa: sidecars antiguos pueden traer mutaciones de tamaño.
        # Limpiamos para fetch siempre el original a máxima resolución.
        poster_url = _strip_shopify_thumb(poster_url)
        if force:
            for stem in _POSTER_STEMS:
                for ext in _POSTER_EXTS:
                    for candidate in (
                        folder / f"{stem}.{ext}",
                        folder / f"{stem}.{ext.upper()}",
                    ):
                        if candidate.exists():
                            try:
                                candidate.unlink()
                            except OSError as exc:
                                log.warning(
                                    "could not remove existing poster %s: %s",
                                    candidate, exc,
                                )
        elif _has_local_poster(folder):
            return None
        try:
            async with httpx.AsyncClient(
                timeout=DEFAULT_TIMEOUT, follow_redirects=True
            ) as client:
                r = await client.get(poster_url)
                if r.status_code >= 400:
                    log.warning(
                        "poster download HTTP %d for %s",
                        r.status_code, poster_url,
                    )
                    return None
                content_type = (r.headers.get("content-type") or "").lower()
                ext = "jpg"
                for known_ext, ct in (
                    ("png", "image/png"),
                    ("webp", "image/webp"),
                    ("jpg", "image/jpeg"),
                ):
                    if ct in content_type:
                        ext = known_ext
                        break
                else:
                    # Fallback a la extensión de la URL si el content-type
                    # no es informativo.
                    lower = poster_url.lower().split("?", 1)[0]
                    for known in _POSTER_EXTS:
                        if lower.endswith("." + known):
                            ext = known if known != "jpeg" else "jpg"
                            break
                dest = folder / f"poster.{ext}"
                tmp = folder / f"poster.{ext}.tmp"
                raw = r.content
                tmp.write_bytes(raw)
                os.replace(tmp, dest)
                # Recorta el padding negro que BJJFanatics mete en muchos
                # posters portrait. Si el fichero ya está ajustado, no-op.
                _trim_black_borders(dest)
                log.info("downloaded poster to %s", dest)
                return dest.name
        except (httpx.HTTPError, OSError) as exc:
            log.warning("poster download failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Endpoints (mantienen el comportamiento exacto del legacy)
    # ------------------------------------------------------------------

    async def list_providers(self) -> Any:
        """Proxy GET a ``chapter-splitter /oracle/providers``."""
        url = f"{self._splitter_url}/oracle/providers"
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                r = await client.get(url)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"backend unreachable: {exc}") from exc
        if r.status_code >= 400:
            raise UpstreamError(
                f"backend error {r.status_code}: {r.text}",
                status_code=502,
            )
        try:
            return r.json()
        except ValueError as exc:
            raise UpstreamError("backend returned invalid JSON") from exc

    def get_oracle(self, instructional_path: str) -> dict[str, Any]:
        """Devuelve el oracle cacheado en el sidecar."""
        folder = self._resolve_instructional(instructional_path)
        meta = _read_meta(folder)
        oracle = meta.get("oracle")
        if not oracle:
            raise NotFoundError("no oracle cached")
        return oracle

    async def resolve(
        self,
        instructional_path: str,
        body: dict[str, Any],
    ) -> Any:
        """Proxy ``/oracle/search`` con title/author derivados del folder."""
        folder = self._resolve_instructional(instructional_path)
        provider_id = body.get("provider_id")  # puede ser None (autodetect)
        override_title = body.get("title")
        override_author = body.get("author")

        meta = _read_meta(folder)
        title, author = _derive_title_author(folder, meta)
        if isinstance(override_title, str) and override_title.strip():
            title = override_title.strip()
        if isinstance(override_author, str) and override_author.strip():
            author = override_author.strip()

        payload: dict[str, Any] = {"title": title, "author": author}
        if provider_id is not None:
            payload["provider_id"] = provider_id

        url = f"{self._splitter_url}/oracle/search"
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                r = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"backend unreachable: {exc}") from exc
        if r.status_code >= 400:
            raise UpstreamError(
                f"backend error {r.status_code}: {r.text}",
                status_code=502,
            )
        try:
            return r.json()
        except ValueError as exc:
            raise UpstreamError("backend returned invalid JSON") from exc

    async def scrape(
        self,
        instructional_path: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Scrape vía backend + persiste en sidecar + auto-poster."""
        folder = self._resolve_instructional(instructional_path)
        if (
            not isinstance(body, dict)
            or not isinstance(body.get("url"), str)
            or not body["url"]
        ):
            raise _InvalidOracle("body must include 'url' string")

        target_url = body["url"]
        backend_url = f"{self._splitter_url}/oracle/scrape"
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                r = await client.post(backend_url, json={"url": target_url})
        except httpx.HTTPError as exc:
            raise UpstreamError(f"backend unreachable: {exc}") from exc
        if r.status_code >= 400:
            raise UpstreamError(
                f"backend error {r.status_code}: {r.text}",
                status_code=502,
            )
        try:
            oracle_result = r.json()
        except ValueError as exc:
            raise UpstreamError("backend returned invalid JSON") from exc

        validated = _validate_oracle_result(oracle_result)

        meta = _read_meta(folder)
        meta["oracle"] = validated
        meta["url_bjjfanatics"] = target_url
        _write_meta_atomic(folder, meta)

        saved = await self._download_poster_if_missing(
            folder, validated.get("poster_url")
        )
        response = dict(validated)
        if saved:
            response["poster_downloaded"] = saved
            try:
                cache = self._scan_cache_loader()
                self._patch_poster(cache, folder.name, saved)
            except Exception:  # noqa: BLE001
                log.warning(
                    "patch_poster_in_cache failed for %s",
                    folder.name, exc_info=True,
                )
        return response

    async def put_oracle(
        self,
        instructional_path: str,
        body: Any,
    ) -> dict[str, Any]:
        """Edición manual del oracle (sin pasar por scrape backend)."""
        folder = self._resolve_instructional(instructional_path)
        validated = _validate_oracle_result(body)

        meta = _read_meta(folder)
        meta["oracle"] = validated
        if validated.get("product_url"):
            meta["url_bjjfanatics"] = validated["product_url"]
        _write_meta_atomic(folder, meta)

        saved = await self._download_poster_if_missing(
            folder, validated.get("poster_url")
        )
        response = dict(validated)
        if saved:
            response["poster_downloaded"] = saved
            try:
                cache = self._scan_cache_loader()
                self._patch_poster(cache, folder.name, saved)
            except Exception:  # noqa: BLE001
                log.warning(
                    "patch_poster_in_cache failed for %s",
                    folder.name, exc_info=True,
                )
        return response

    def delete_oracle(self, instructional_path: str) -> dict[str, Any]:
        """Elimina la sección ``oracle`` del sidecar (preserva el resto)."""
        folder = self._resolve_instructional(instructional_path)
        meta = _read_meta(folder)
        if "oracle" in meta:
            meta.pop("oracle", None)
            _write_meta_atomic(folder, meta)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Re-export para callers fuera del módulo (legacy ``api.app`` usa
    # ``_download_poster_if_missing`` y ``SIDECAR_NAME`` directamente
    # desde ``api.oracle`` en el endpoint de redownload de poster).
    # ------------------------------------------------------------------

    async def download_poster(
        self,
        folder: Path,
        poster_url: Optional[str],
        *,
        force: bool = False,
    ) -> Optional[str]:
        """Wrapper público sobre ``_download_poster_if_missing``.

        Mantiene la firma esperada por ``api/app.py`` para el endpoint
        ``POST /api/library/{name}/poster/redownload``.
        """
        return await self._download_poster_if_missing(
            folder, poster_url, force=force
        )
