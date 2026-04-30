"""Servicio principal del módulo library.

Encapsula:
* Walk de la biblioteca (``scan``) para construir el listado de
  instructionals con flags por vídeo.
* Lectura del listado cacheado (``get_cached``) y refresh background.
* Detalle de un instructional (``get_detail``) agrupando vídeos por
  season + lazy-fill de duraciones.
* Re-discover de un único instructional (``refresh_instructional``).
* Servir/subir/redescargar poster.

Recibe ``LibraryCache``, ``library_path_loader`` (función) y
``poster_downloader`` por DI. ``BackgroundJobsService`` no se usa aquí
— las refrescos son fire-and-forget en el executor del loop, no son
"jobs" en el sentido del módulo jobs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .cache import (
    LibraryCache,
    enrich_with_poster,
    find_poster,
    find_poster_cached,
    patch_poster_in_cache,
)
from .refresh import ensure_duration, rediscover_instructional

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}
SIDECAR_NAME = ".bjj-meta.json"

_SEASON_RE = re.compile(r"(?:Season|Volume|Vol)\s*(\d+)", re.IGNORECASE)
_EPISODE_RE = re.compile(r"\bS(\d{1,2})E\d{1,3}\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Funciones puras (auxiliares de scan_library)
# ---------------------------------------------------------------------------


def season_from_path(video_path: str, instructional_path: str) -> str:
    """Deriva la season del vídeo a partir de su path relativo al instructional.

    Prioridades:
    1. Segmento de carpeta que coincide con "Season N", "Volume N" o "Vol N".
    2. Código de episodio "SNNeMMM" en el nombre del fichero.

    Devuelve ``"Sin temporada"`` si nada coincide.
    """
    if not video_path:
        return "Sin temporada"
    rel = video_path
    try:
        inst_norm = instructional_path.replace("\\", "/").rstrip("/")
        vid_norm = video_path.replace("\\", "/")
        if inst_norm and vid_norm.lower().startswith(inst_norm.lower() + "/"):
            rel = vid_norm[len(inst_norm) + 1 :]
    except Exception:  # pragma: no cover
        pass
    m = _SEASON_RE.search(rel)
    if m:
        return f"Season {int(m.group(1))}"
    m = _EPISODE_RE.search(rel)
    if m:
        return f"Season {int(m.group(1))}"
    return "Sin temporada"


def scan_library(root_path: str) -> list[dict]:
    """Walk del árbol de directorios para construir el listado.

    Lógica idéntica al ``scan_library`` legacy de ``app.py``: salta carpetas
    de artefactos (``elevenlabs/``, ``doblajes/``), filtra ``_DOBLADO`` por
    nombre, agrupa por instructional usando "Season N" como separador.
    """
    root = Path(root_path)
    if not root.exists():
        return []

    instructionals: dict[str, dict] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        # In-place mutation prunes os.walk descent.
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in ("elevenlabs", "doblajes")
        ]

        dp = Path(dirpath)
        videos = sorted(
            f for f in filenames
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS
            and "_DOBLADO" not in Path(f).stem
        )
        if not videos:
            continue

        # Determinar nombre + carpeta raíz del instructional.
        folder_name = dp.name
        if "season" in folder_name.lower():
            instr_name = dp.parent.name
            instr_path = dp.parent
        else:
            instr_name = folder_name
            instr_path = dp

        if instr_name not in instructionals:
            sidecar = instr_path / SIDECAR_NAME
            author = ""
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    author = meta.get("instructor", "") or ""
                except (OSError, ValueError):
                    pass
            instructionals[instr_name] = {
                "name": instr_name,
                "path": str(instr_path),
                "author": author,
                "videos": [],
                "total_videos": 0,
                "chapters_detected": 0,
                "subtitled": 0,
                "dubbed": 0,
            }

        for vf in videos:
            vpath = dp / vf
            base = vpath.stem

            has_srt = (dp / f"{base}.en.srt").exists() or (dp / f"{base}.srt").exists()
            has_es_srt = (
                (dp / f"{base}.es.srt").exists()
                or (dp / f"{base}.ES.srt").exists()
                or (dp / f"{base}_ES.srt").exists()
                or (dp / f"{base}_ESP_DUB.srt").exists()
            )
            has_dubbed = (
                any((dp / f"{base}{sfx}").exists() for sfx in ["_DOBLADO.mkv", "_DOBLADO.mp4"])
                or (dp / "doblajes" / f"{base}.mkv").exists()
                or (dp / "elevenlabs" / vf).exists()
            )
            is_chapter = bool(re.search(r"S\d{2}E\d{2}", vf))

            try:
                size_mb = round(vpath.stat().st_size / (1024 * 1024), 1)
            except OSError:
                size_mb = None
            instructionals[instr_name]["videos"].append({
                "filename": vf,
                "path": str(vpath),
                "size_mb": size_mb,
                "duration": None,
                "has_subtitles_en": has_srt,
                "has_subtitles_es": has_es_srt,
                "has_dubbing": has_dubbed,
                "is_chapter": is_chapter,
            })
            instructionals[instr_name]["total_videos"] += 1
            if has_srt:
                instructionals[instr_name]["subtitled"] += 1
            if has_dubbed:
                instructionals[instr_name]["dubbed"] += 1
            if is_chapter:
                instructionals[instr_name]["chapters_detected"] += 1

    return list(instructionals.values())


# ---------------------------------------------------------------------------
# Service (estado en instancia)
# ---------------------------------------------------------------------------


class _Forbidden(Exception):
    """Anti-traversal: el path resuelto está fuera de library_path."""


class _NotFound(Exception):
    """Recurso no existe (instructional, poster, sidecar, cache)."""


# Tipo del downloader de poster: recibe (target_dir, url, force=...) y devuelve
# el filename guardado o None. Lo provee scrapper.service.
PosterDownloader = Callable[..., Awaitable[Optional[str]]]


class LibraryService:
    """Operaciones públicas de la biblioteca de instructionals.

    Estado de instancia (sustituye a los globales antiguos de app.py):
    * ``_refresh_inflight`` — flag de coalescing del background refresh.
    * El cache (``LibraryCache``) y el library_path_loader vienen por DI.
    """

    def __init__(
        self,
        cache: LibraryCache,
        library_path_loader: Callable[[], Optional[str]],
        poster_downloader: Optional[PosterDownloader] = None,
    ) -> None:
        self._cache = cache
        self._library_path_loader = library_path_loader
        self._poster_downloader = poster_downloader
        self._refresh_lock = asyncio.Lock()
        self._refresh_inflight = False

    # --- Helpers internos ------------------------------------------------

    def _resolve_under_library(self, name: str) -> Path:
        """Resuelve ``library_path/name`` validando contra path-traversal.

        Lanza ``_NotFound`` si library_path no está configurado o si el
        target no existe; ``_Forbidden`` si escapa de la librería.
        """
        lib = self._library_path_loader()
        if not lib:
            raise _NotFound("library_path not configured")
        base = Path(lib).resolve()
        try:
            target = (base / name).resolve()
        except OSError as exc:
            raise _Forbidden(f"invalid path: {exc}") from exc
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise _Forbidden("path traversal denied") from exc
        if target == base:
            raise _Forbidden("invalid target")
        if not target.exists() or not target.is_dir():
            raise _NotFound("instructional not found")
        return target

    # --- scan ------------------------------------------------------------

    async def scan(self, root_path: Optional[str] = None) -> dict[str, Any]:
        """Lanza un escaneo completo y persiste el resultado en cache.

        Si no se proporciona ``root_path``, se resuelve desde
        ``library_path_loader``. Lanza ``ValueError`` si no hay path
        configurado, ``_NotFound`` si el path no existe.
        """
        path = root_path or self._library_path_loader()
        if not path:
            raise ValueError(
                "Library path not configured. Provide 'path' in the request "
                "or configure it in Settings."
            )
        if not Path(path).exists():
            raise _NotFound(
                f"Path not accessible: {path}. Verify the path exists and "
                "the server has read permissions."
            )

        def _scan_sync() -> list[dict]:
            lib = scan_library(path)
            return enrich_with_poster(lib)

        library = await asyncio.get_event_loop().run_in_executor(None, _scan_sync)
        try:
            self._cache.save(library)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to persist library cache: %s", exc)
        return {"instructionals": library}

    # --- get cached + background refresh --------------------------------

    def get_cached(self, *, refresh: bool = False) -> dict[str, Any]:
        """Devuelve el listado cacheado. Si ``refresh=True`` lanza un
        rescan en background sin bloquear la response.

        Si la cache está vacía (cold), lanza también un refresh para que
        el siguiente poll tenga datos.
        """
        if refresh:
            self._kick_background_refresh()

        data = self._cache.load()
        if data is None:
            self._kick_background_refresh()
            return {"instructionals": [], "refreshing": True}
        if isinstance(data, dict):
            return {**data, "refreshing": self._refresh_inflight}
        return data

    def _kick_background_refresh(self) -> None:
        """Fire-and-forget: rescan en executor con coalescing por flag.

        Mismo patrón que el legacy ``_kick_background_library_refresh`` de
        ``app.py`` — un solo refresh en vuelo a la vez para no encolar N
        scans al recargar la página.
        """
        if self._refresh_inflight:
            return
        path = self._library_path_loader()
        if not path or not Path(path).exists():
            return

        self._refresh_inflight = True

        def _scan_sync() -> None:
            try:
                lib = scan_library(path)
                lib = enrich_with_poster(lib)
                try:
                    self._cache.save(lib)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to persist library cache: %s", exc)
            except Exception as exc:  # noqa: BLE001
                log.warning("background library refresh failed: %s", exc)
            finally:
                self._refresh_inflight = False

        asyncio.get_event_loop().run_in_executor(None, _scan_sync)

    # --- detalle de un instructional ------------------------------------

    async def get_detail(self, name: str, *, refresh: bool = True) -> dict[str, Any]:
        """Devuelve el detalle de un instructional con vídeos agrupados por season.

        Si ``refresh=True`` (default), re-stata sidecars y rellena duraciones
        en background sin bloquear.

        Lanza ``_NotFound`` si la cache está vacía o el instructional no existe.
        """
        data = self._cache.load()
        if data is None:
            raise _NotFound("no scan cache")

        items = data.get("instructionals", []) if isinstance(data, dict) else []
        match = next((it for it in items if it.get("name") == name), None)
        if match is None:
            raise _NotFound("instructional not found")

        if refresh:
            def _refresh_and_backfill() -> None:
                rediscover_instructional(match)
                for v in (match.get("videos") or []):
                    if isinstance(v, dict):
                        ensure_duration(v)
                try:
                    self._cache.save(items)
                except Exception:  # noqa: BLE001
                    pass

            asyncio.get_event_loop().run_in_executor(None, _refresh_and_backfill)

        inst_path = match.get("path", "")
        raw_videos = match.get("videos", []) or []
        videos: list[dict[str, Any]] = []
        for v in raw_videos:
            if not isinstance(v, dict):
                continue
            vp = v.get("path", "")
            videos.append({
                "path": vp,
                "filename": v.get("filename") or (Path(vp).name if vp else ""),
                "season": season_from_path(vp, inst_path),
                "size": v.get("size"),
                "duration": v.get("duration"),
                "is_chapter": v.get("is_chapter", False),
                "has_subtitles_en": v.get("has_subtitles_en", False),
                "has_subtitles_es": v.get("has_subtitles_es", False),
                "has_dubbing": v.get("has_dubbing", False),
            })

        return {
            "name": match.get("name"),
            "path": inst_path,
            "has_poster": bool(match.get("has_poster")),
            "poster_filename": match.get("poster_filename"),
            "poster_mtime": match.get("poster_mtime"),
            "videos": videos,
        }

    # --- refresh de un instructional ------------------------------------

    async def refresh_instructional(self, name: str) -> dict[str, Any]:
        """Re-discover ligero de un instructional.

        Más barato que un ``/api/scan`` completo: solo walkea esa carpeta y
        preserva campos cacheados (duration, etc.) para vídeos que sigan
        existiendo.
        """
        data = self._cache.load()
        if data is None or not isinstance(data, dict):
            raise _NotFound("no scan cache")
        items = data.get("instructionals") or []
        match = next((it for it in items if it.get("name") == name), None)
        if match is None:
            raise _NotFound("instructional not found")

        def _rediscover_sync() -> None:
            rediscover_instructional(match)
            try:
                self._cache.save(items)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to persist refreshed cache: %s", exc)

        await asyncio.get_event_loop().run_in_executor(None, _rediscover_sync)
        return {"ok": True, "videos": len(match.get("videos") or [])}

    # --- poster ---------------------------------------------------------

    def find_cached_poster(self, name: str) -> tuple[Path, Optional[str]]:
        """Resuelve el path del poster y el filename cacheado.

        Devuelve ``(target_dir, poster_path_o_None)``. Lanza ``_NotFound`` /
        ``_Forbidden`` según corresponda.
        """
        target = self._resolve_under_library(name)

        cached_poster_filename: Optional[str] = None
        try:
            cache_data = self._cache.load()
            if cache_data and isinstance(cache_data, dict):
                for item in cache_data.get("instructionals", []) or []:
                    if item.get("name") == name:
                        cached_poster_filename = item.get("poster_filename")
                        break
        except Exception:  # noqa: BLE001
            cached_poster_filename = None

        poster = find_poster_cached(target, cached_poster_filename)
        if poster is None:
            raise _NotFound("poster not found")
        return target, poster

    def upload_poster(self, name: str, ext: str, contents: bytes) -> dict[str, Any]:
        """Guarda un poster custom en el folder del instructional.

        Reemplaza posters canónicos preexistentes (poster.* / cover.*) con
        otra extensión.
        """
        allowed_ext = {"jpg", "jpeg", "png", "webp"}
        if ext.lower() not in allowed_ext:
            raise ValueError(f"unsupported extension: {ext}")
        if len(contents) > 10 * 1024 * 1024:
            raise ValueError("file too large (max 10MB)")

        target = self._resolve_under_library(name)

        # Limpiar posters canónicos preexistentes con otra extensión.
        for stem in ("poster", "cover"):
            for old_ext in allowed_ext:
                old = target / f"{stem}.{old_ext}"
                if old.exists():
                    try:
                        old.unlink()
                    except OSError:
                        pass

        dest = target / f"poster.{ext}"
        dest.write_bytes(contents)
        patch_poster_in_cache(self._cache, name, dest.name)
        return {"saved": dest.name, "size": len(contents)}

    async def redownload_poster(self, name: str) -> dict[str, Any]:
        """Re-descarga el poster desde la URL del sidecar oracle.

        Requiere que ``self._poster_downloader`` esté inyectado (viene de
        ``scrapper.service``). Lanza ``_NotFound`` si no hay sidecar o
        ``poster_url``; ``RuntimeError`` si el downloader devuelve None.
        """
        if self._poster_downloader is None:
            raise RuntimeError("poster_downloader not configured")

        target = self._resolve_under_library(name)
        sidecar = target / SIDECAR_NAME
        if not sidecar.exists():
            raise _NotFound("no oracle sidecar")
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid sidecar: {exc}") from exc

        poster_url = (meta.get("oracle") or {}).get("poster_url")
        if not poster_url:
            raise _NotFound("no poster_url in oracle")

        saved = await self._poster_downloader(target, poster_url, force=True)
        if not saved:
            raise RuntimeError("poster download failed")
        patch_poster_in_cache(self._cache, name, saved)
        return {"saved": saved}
