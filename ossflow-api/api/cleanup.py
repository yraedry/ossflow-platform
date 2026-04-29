"""Auto-limpieza de artefactos viejos en la biblioteca BJJ.

Escanea una ruta bajo ``library_path`` y clasifica candidatos a borrar:
- ``orphan_srt``: subtítulos sin vídeo hermano (mismo stem .mkv/.mp4)
- ``old_dubbed``: ficheros ``*_DOBLADO.{mkv,mp4}`` más antiguos que el .ES.srt hermano
- ``temp_files``: ``.tmp``, ``.part``, ``.crdownload``, ``~*``, ``*.bak``
- ``empty_dirs``: directorios vacíos

Las operaciones destructivas se hacen desde ``POST /apply`` y SIEMPRE revalidan
que cada path esté bajo ``library_path`` (anti path-traversal).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

import asyncio

from api.background_jobs import registry as _jobs_registry
from api.settings import get_library_path

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cleanup", tags=["cleanup"])


VIDEO_EXTS = {".mkv", ".mp4"}
TEMP_EXTS = {".tmp", ".part", ".crdownload", ".bak"}


def _resolve_under_library(raw_path: str) -> Path:
    """Valida que ``raw_path`` resuelva dentro del ``library_path`` configurado.

    Lanza ``HTTPException(400)`` si library_path no está configurado o el path
    queda fuera (anti traversal). Devuelve el Path resuelto (absoluto).
    """
    lib = get_library_path()
    if not lib:
        raise HTTPException(status_code=400, detail="library_path no está configurado")
    try:
        lib_resolved = Path(lib).resolve(strict=False)
        target = Path(raw_path).resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Path inválido: {exc}") from exc

    try:
        target.relative_to(lib_resolved)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail=f"Path fuera de library_path: {raw_path}",
        )
    return target


def _is_temp_file(name: str) -> bool:
    lower = name.lower()
    if lower.startswith("~"):
        return True
    ext = os.path.splitext(lower)[1]
    return ext in TEMP_EXTS


def _safe_stat(p: Path) -> tuple[int, float] | None:
    try:
        st = p.stat()
        return st.st_size, st.st_mtime
    except OSError:
        return None


def _info(p: Path) -> dict[str, Any] | None:
    s = _safe_stat(p)
    if s is None:
        return None
    size, mtime = s
    return {"path": str(p), "size": int(size), "mtime": float(mtime)}


def _scan_tree(root: Path) -> dict[str, Any]:
    orphan_srt: list[dict[str, Any]] = []
    old_dubbed: list[dict[str, Any]] = []
    temp_files: list[dict[str, Any]] = []
    empty_dirs: list[dict[str, Any]] = []

    if not root.exists():
        raise HTTPException(status_code=404, detail=f"Path no existe: {root}")

    for dirpath, dirnames, filenames in os.walk(root):
        dir_p = Path(dirpath)
        names_set = set(filenames)

        # Empty dirs (no files and no subdirs)
        if not filenames and not dirnames:
            info = _info(dir_p)
            if info:
                empty_dirs.append(info)

        # Index stems of videos in this dir
        video_stems = {
            Path(f).stem for f in filenames
            if Path(f).suffix.lower() in VIDEO_EXTS
        }

        for fname in filenames:
            fpath = dir_p / fname
            lower = fname.lower()
            suffix = Path(fname).suffix.lower()

            # temp files
            if _is_temp_file(fname):
                info = _info(fpath)
                if info:
                    temp_files.append(info)
                continue

            # orphan .srt
            if suffix == ".srt":
                stem = Path(fname).stem
                # .ES.srt or other suffixes: match against any video stem that is
                # a prefix (so ``video.ES`` should pair with ``video.mkv``).
                base_stem = stem
                if "." in stem:
                    base_stem = stem.split(".")[0]
                if stem not in video_stems and base_stem not in video_stems:
                    info = _info(fpath)
                    if info:
                        orphan_srt.append(info)

            # old dubbed
            if suffix in VIDEO_EXTS and Path(fname).stem.endswith("_DOBLADO"):
                # Buscar .ES.srt hermano cuyo stem sea el del video SIN _DOBLADO
                orig_stem = Path(fname).stem[: -len("_DOBLADO")]
                es_srt = dir_p / f"{orig_stem}.es.srt"
                if not es_srt.exists():
                    es_srt = dir_p / f"{orig_stem}.ES.srt"
                if es_srt.exists():
                    fstat = _safe_stat(fpath)
                    sstat = _safe_stat(es_srt)
                    if fstat and sstat and fstat[1] < sstat[1]:
                        info = _info(fpath)
                        if info:
                            old_dubbed.append(info)

    # sort each by size desc
    for lst in (orphan_srt, old_dubbed, temp_files, empty_dirs):
        lst.sort(key=lambda x: x["size"], reverse=True)

    total_items = len(orphan_srt) + len(old_dubbed) + len(temp_files) + len(empty_dirs)
    total_bytes = sum(
        i["size"] for i in (*orphan_srt, *old_dubbed, *temp_files, *empty_dirs)
    )

    return {
        "categories": {
            "orphan_srt": orphan_srt,
            "old_dubbed": old_dubbed,
            "temp_files": temp_files,
            "empty_dirs": empty_dirs,
        },
        "total_bytes": total_bytes,
        "total_items": total_items,
    }


@router.get("/scan")
async def scan(path: str):
    """Escanea ``path`` y devuelve candidatos a borrar agrupados por categoría."""
    if not path:
        raise HTTPException(status_code=400, detail="path es obligatorio")
    target = _resolve_under_library(path)
    return _scan_tree(target)


@router.post("/start")
async def start_scan(path: str):
    """Launch a cleanup scan as an async background job.

    Returns ``{job_id}``. Poll ``/api/background-jobs/{job_id}`` or
    ``/api/cleanup/job/{job_id}`` for progress/result.
    """
    if not path:
        raise HTTPException(status_code=400, detail="path es obligatorio")
    target = _resolve_under_library(path)

    async def _coro(update_progress):
        update_progress(0.0, f"Scanning {target}...")

        def _work():
            return _scan_tree(target)

        update_progress(10.0, "Walking filesystem...")
        result = await asyncio.to_thread(_work)
        update_progress(
            100.0,
            f"Found {result.get('total_items', 0)} items",
        )
        return result

    job = _jobs_registry.submit("cleanup_scan", _coro, {"path": str(target)})
    return {"job_id": job.id}


@router.get("/job/{job_id}")
async def get_job(job_id: str):
    job = _jobs_registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.post("/apply")
async def apply(request: Request):
    """Borra los paths indicados (revalidando cada uno bajo library_path).

    Body JSON: ``{"paths": [str], "dry_run": bool}`` (default ``dry_run=false``).
    Un item que falla NO aborta el resto; los errores se recogen en ``errors``.
    Nunca borra vídeos ``.mkv``/``.mp4`` que no fueran clasificados como
    ``old_dubbed`` (heurística: deben terminar en ``_DOBLADO.{mkv,mp4}``).
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "Body debe ser JSON"}, status_code=422)

    if not isinstance(body, dict):
        return JSONResponse({"detail": "Body debe ser un objeto JSON"}, status_code=422)

    paths = body.get("paths") or []
    dry_run = bool(body.get("dry_run", False))
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        return JSONResponse(
            {"detail": "paths debe ser lista de strings"}, status_code=422
        )

    deleted: list[str] = []
    errors: list[dict[str, str]] = []
    freed_bytes = 0

    for raw in paths:
        try:
            target = _resolve_under_library(raw)
        except HTTPException as e:
            errors.append({"path": raw, "error": str(e.detail)})
            continue

        if not target.exists():
            errors.append({"path": raw, "error": "no existe"})
            continue

        # Protección: no borrar vídeos que no sean *_DOBLADO.*
        suffix = target.suffix.lower()
        if target.is_file() and suffix in VIDEO_EXTS:
            if not target.stem.endswith("_DOBLADO"):
                errors.append(
                    {"path": raw, "error": "vídeo no doblado: borrado denegado"}
                )
                continue

        try:
            if target.is_dir():
                # solo si vacío
                if any(target.iterdir()):
                    errors.append({"path": raw, "error": "directorio no vacío"})
                    continue
                if not dry_run:
                    target.rmdir()
                deleted.append(str(target))
            else:
                st = _safe_stat(target)
                size = st[0] if st else 0
                if not dry_run:
                    target.unlink()
                deleted.append(str(target))
                freed_bytes += int(size)
        except OSError as exc:
            errors.append({"path": raw, "error": str(exc)})

    return {
        "deleted": deleted,
        "errors": errors,
        "freed_bytes": int(freed_bytes),
        "dry_run": dry_run,
    }
