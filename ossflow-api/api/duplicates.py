"""Duplicate video detection.

Heuristic approach (fast path): two videos are candidate duplicates when they
share the same ``(size_bytes, duration_seconds_rounded_to_1s)`` signature.

Optional deep mode confirms candidates by also comparing an md5 of the first
10 MB of each file (flag ``deep=true``).

Single responsibility: walk a subtree under the configured library, group
videos by signature, return duplicates.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException

from api.background_jobs import registry as _jobs_registry
from api.settings import get_library_path

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/duplicates", tags=["duplicates"])

_VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")
_DEEP_SAMPLE_BYTES = 10 * 1024 * 1024  # 10 MB


def _is_under(child: Path, parent: Path) -> bool:
    """Return True if ``child`` resolves under ``parent`` (anti-traversal)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _partial_md5(path: Path, nbytes: int = _DEEP_SAMPLE_BYTES) -> str | None:
    try:
        h = hashlib.md5()
        with path.open("rb") as fh:
            chunk = fh.read(nbytes)
            h.update(chunk)
        return h.hexdigest()
    except Exception as exc:
        log.warning("md5 failed for %s: %s", path, exc)
        return None


def _validate_path(path: str) -> Path:
    lib = get_library_path()
    if not lib:
        raise HTTPException(status_code=400, detail="library_path no configurado")
    root = Path(path)
    if not _is_under(root, Path(lib)):
        raise HTTPException(status_code=403, detail="Path fuera de la librería")
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=404, detail="Path no existe")
    return root


def _scan_duplicates(
    root: Path,
    deep: bool = False,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Core walk+group logic, reused by sync and async entrypoints."""
    from api.app import get_video_info  # lazy so tests can monkeypatch

    # Collect candidate files first so we can report progress
    candidates: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(_VIDEO_EXTS):
                candidates.append(Path(dirpath) / fname)

    total_candidates = len(candidates)
    if progress_cb:
        progress_cb(5.0, f"Found {total_candidates} video candidates")

    signatures: dict[tuple[int, int], list[dict[str, Any]]] = {}
    total_videos = 0

    for idx, fpath in enumerate(candidates):
        try:
            size = fpath.stat().st_size
        except OSError as exc:
            log.warning("stat failed for %s: %s", fpath, exc)
            continue
        try:
            info = get_video_info(str(fpath))
        except Exception as exc:
            log.warning("ffprobe failed for %s: %s", fpath, exc)
            continue
        duration = float(info.get("duration", 0) or 0)
        if duration <= 0:
            continue
        total_videos += 1
        key = (size, int(round(duration)))
        signatures.setdefault(key, []).append({
            "path": str(fpath),
            "size": size,
            "duration_sec": int(round(duration)),
        })
        if progress_cb and total_candidates:
            pct = 5.0 + 85.0 * ((idx + 1) / total_candidates)
            progress_cb(pct, f"Probed {idx + 1}/{total_candidates}")

    groups = [g for g in signatures.values() if len(g) >= 2]

    if deep and groups:
        if progress_cb:
            progress_cb(92.0, "Computing partial md5 for candidate groups")
        confirmed: list[list[dict[str, Any]]] = []
        for group in groups:
            by_hash: dict[str, list[dict[str, Any]]] = {}
            for entry in group:
                digest = _partial_md5(Path(entry["path"]))
                if digest is None:
                    continue
                by_hash.setdefault(digest, []).append(entry)
            for entries in by_hash.values():
                if len(entries) >= 2:
                    confirmed.append(entries)
        groups = confirmed

    wasted_bytes = 0
    for group in groups:
        wasted_bytes += sum(e["size"] for e in group[1:])

    if progress_cb:
        progress_cb(100.0, f"Done: {len(groups)} duplicate groups")

    return {
        "groups": groups,
        "stats": {
            "total_videos": total_videos,
            "groups_found": len(groups),
            "wasted_bytes": wasted_bytes,
        },
    }


@router.get("/scan")
async def scan(path: str, deep: bool = False) -> dict[str, Any]:
    """Walk ``path`` and return groups of duplicate-candidate videos.

    Response shape::
        {
          "groups": [ [ {path, size, duration_sec}, ... ], ... ],
          "stats":  { "total_videos", "groups_found", "wasted_bytes" }
        }
    Only groups with >= 2 entries are returned.
    """
    root = _validate_path(path)
    return _scan_duplicates(root, deep=deep)


@router.post("/start")
async def start_scan(path: str, deep: bool = False):
    """Launch a duplicates scan as an async background job."""
    root = _validate_path(path)

    async def _coro(update_progress):
        update_progress(0.0, f"Scanning {root} for duplicates...")

        def _work():
            return _scan_duplicates(
                root,
                deep=deep,
                progress_cb=lambda p, m: update_progress(p, m),
            )

        return await asyncio.to_thread(_work)

    job = _jobs_registry.submit(
        "duplicates_scan", _coro, {"path": str(root), "deep": bool(deep)}
    )
    return {"job_id": job.id}


@router.get("/job/{job_id}")
async def get_job(job_id: str):
    job = _jobs_registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()
