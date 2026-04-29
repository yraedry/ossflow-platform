"""Burn Spanish subtitles (.ES.srt) into video files with ffmpeg.

Single responsibility: given a video file or a Season folder, produce
``<stem>_SUB_ES.mp4`` next to each source video whose matching ``.ES.srt``
sidecar exists. CPU-bound; runs as a background job so the caller returns
immediately and can poll ``/api/background-jobs/{id}``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.background_jobs import registry
from api.settings import get_library_path

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/burn-subs", tags=["burn-subs"])

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
SUB_SUFFIXES = [".es.srt", ".ES.srt", "_ES.srt", "_ESP_DUB.srt"]
OUT_SUFFIX = "_SUB_ES.mp4"


def _resolve_within_library(candidate: Path, library_root: Path) -> Path:
    try:
        resolved = candidate.resolve(strict=False)
        root_resolved = library_root.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise HTTPException(status_code=403, detail=f"Path error: {e}")
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise HTTPException(
            status_code=403, detail="Path escapes library_path"
        )
    return resolved


def _collect_targets(root: Path) -> list[tuple[Path, Path]]:
    """Return list of (video, srt) pairs to burn.

    If ``root`` is a file, wrap it. If directory, scan non-recursively for
    videos with a matching ``.ES.srt`` sidecar, skipping already-burned
    outputs.
    """
    pairs: list[tuple[Path, Path]] = []
    candidates: list[Path]
    if root.is_file():
        candidates = [root]
    else:
        candidates = [p for p in root.iterdir() if p.is_file()]

    for video in candidates:
        if video.suffix.lower() not in VIDEO_EXTS:
            continue
        if video.name.endswith(OUT_SUFFIX):
            continue
        srt = next(
            (video.with_name(video.stem + s) for s in SUB_SUFFIXES
             if video.with_name(video.stem + s).exists()),
            None,
        )
        if srt is None:
            continue
        out = video.with_name(video.stem + OUT_SUFFIX)
        if out.exists():
            continue
        pairs.append((video, srt))
    return pairs


def _ffmpeg_escape_subs_path(p: Path) -> str:
    """Escape a path for the ffmpeg ``subtitles=`` filter.

    Windows drive letters and backslashes need escaping inside the filter
    graph. Forward slashes + escaping the colon works cross-platform.
    """
    s = str(p).replace("\\", "/")
    # Escape drive colon: C:/foo -> C\:/foo
    if len(s) >= 2 and s[1] == ":":
        s = s[0] + r"\:" + s[2:]
    return s


# Límite de encode por vídeo. En H.264 veryfast, 6 h de vídeo son raros; si se
# supera, casi seguro es un vídeo corrupto que tiene a ffmpeg colgado.
_BURN_TIMEOUT_SEC = 6 * 60 * 60


async def _burn_one(video: Path, srt: Path) -> tuple[bool, str]:
    out = video.with_name(video.stem + OUT_SUFFIX)
    tmp = out.with_suffix(out.suffix + ".part")
    vf = f"subtitles='{_ffmpeg_escape_subs_path(srt)}'"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        vf,
        "-c:a",
        "copy",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        str(tmp),
    ]
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_BURN_TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            # Un ffmpeg colgado bloqueaba el worker indefinidamente — matamos
            # el proceso y liberamos el temporal.
            proc.kill()
            try:
                await proc.wait()
            except Exception:  # noqa: BLE001
                pass
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            return False, f"ffmpeg timeout after {_BURN_TIMEOUT_SEC}s"
        if proc.returncode != 0:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            return False, (stderr.decode(errors="replace") or "ffmpeg failed")[-400:]
        tmp.replace(out)
        return True, str(out)
    except FileNotFoundError:
        return False, "ffmpeg binary not found in PATH"
    except Exception as exc:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False, f"{exc.__class__.__name__}: {exc}"


def _make_coro(targets: list[tuple[Path, Path]]):
    async def _run(update_progress) -> dict:
        total = len(targets)
        done: list[str] = []
        failed: list[dict[str, str]] = []
        for i, (video, srt) in enumerate(targets):
            update_progress(
                (i / total) * 100.0 if total else 0.0,
                f"Quemando {video.name} ({i + 1}/{total})",
            )
            ok, info = await _burn_one(video, srt)
            if ok:
                done.append(info)
            else:
                failed.append({"video": str(video), "error": info})
        update_progress(100.0, f"Completado: {len(done)}/{total}")
        return {
            "burned": done,
            "failed": failed,
            "total": total,
        }

    return _run


@router.post("")
@router.post("/")
async def start_burn(request: Request) -> Any:
    """Start a burn-subs job. Body: ``{"path": "..."}``.

    ``path`` may point to a single video file or a Season folder. Returns
    the created background-job record; poll ``/api/background-jobs/{id}``.
    """
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=503, detail="ffmpeg not available")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Body must be an object")

    raw_path = body.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise HTTPException(status_code=422, detail="path is required")

    lib = get_library_path()
    if not lib:
        raise HTTPException(status_code=400, detail="library_path not configured")

    target = _resolve_within_library(Path(raw_path), Path(lib))
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {raw_path}")

    pairs = _collect_targets(target)
    if not pairs:
        return JSONResponse(
            status_code=409,
            content={
                "error": "No videos with matching ES subtitle sidecar found (.ES.srt / _ES.srt)",
                "path": str(target),
            },
        )

    job = registry.submit(
        type="burn_subs",
        coro_factory=_make_coro(pairs),
        params={"path": str(target), "count": len(pairs)},
    )
    return job.to_dict()


# WIRE_ROUTER: from api.burn_subs import router as burn_subs_router; app.include_router(burn_subs_router)
