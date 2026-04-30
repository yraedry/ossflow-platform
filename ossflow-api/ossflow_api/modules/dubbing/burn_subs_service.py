"""Burn-subs: quema SRT en vídeo con ffmpeg.

Absorbido del antiguo ``api/burn_subs.py`` al módulo ``dubbing`` por
afinidad funcional (es una operación derivada del doblaje: cuando hay
``.ES.srt``, generar ``<stem>_SUB_ES.mp4`` con los subs hardcoded).

Mantiene el contrato HTTP ``POST /api/burn-subs`` para no romper el
frontend. Internamente usa ``BackgroundJobsService`` para encolar el
job (cierre acoplamiento #4 con jobs).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

from ossflow_api.modules.jobs.services.background import BackgroundJobsService

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
SUB_SUFFIXES = [".es.srt", ".ES.srt", "_ES.srt", "_ESP_DUB.srt"]
OUT_SUFFIX = "_SUB_ES.mp4"

# 6 h tope de encode por vídeo. Por encima, casi seguro ffmpeg colgado.
_BURN_TIMEOUT_SEC = 6 * 60 * 60


def _resolve_within_library(candidate: Path, library_root: Path) -> Path:
    """Anti-traversal. Lanza ``PermissionError`` si el path escapa de la lib."""
    try:
        resolved = candidate.resolve(strict=False)
        root_resolved = library_root.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise PermissionError(f"Path error: {exc}") from exc
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PermissionError("Path escapes library_path") from exc
    return resolved


def _collect_targets(root: Path) -> list[tuple[Path, Path]]:
    """Devuelve pares (vídeo, srt) a quemar. Salta los ya-quemados."""
    pairs: list[tuple[Path, Path]] = []
    candidates: list[Path] = (
        [root] if root.is_file() else [p for p in root.iterdir() if p.is_file()]
    )

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
    """Escape de path para el filtro ``subtitles=`` de ffmpeg."""
    s = str(p).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        s = s[0] + r"\:" + s[2:]
    return s


async def _burn_one(video: Path, srt: Path) -> tuple[bool, str]:
    out = video.with_name(video.stem + OUT_SUFFIX)
    tmp = out.with_suffix(out.suffix + ".part")
    vf = f"subtitles='{_ffmpeg_escape_subs_path(srt)}'"
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vf", vf,
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
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


class BurnSubsService:
    """Encola jobs de burn-subs vía ``BackgroundJobsService``."""

    def __init__(
        self,
        jobs: BackgroundJobsService,
        library_path_loader: Callable[[], Optional[str]],
    ) -> None:
        self._jobs = jobs
        self._library_path_loader = library_path_loader

    @staticmethod
    def ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    def submit(self, raw_path: str) -> dict:
        """Resuelve el path, recolecta targets y encola el job.

        Lanza:
        * ``ValueError`` si library_path no está configurado, path no existe,
          o no hay vídeos con SRT hermano.
        * ``PermissionError`` si el path escapa de la librería.
        """
        lib = self._library_path_loader()
        if not lib:
            raise ValueError("library_path not configured")
        target = _resolve_within_library(Path(raw_path), Path(lib))
        if not target.exists():
            raise ValueError(f"Not found: {raw_path}")

        pairs = _collect_targets(target)
        if not pairs:
            raise ValueError(
                "No videos with matching ES subtitle sidecar found "
                "(.ES.srt / _ES.srt)"
            )

        job = self._jobs.submit(
            type="burn_subs",
            coro_factory=_make_coro(pairs),
            params={"path": str(target), "count": len(pairs)},
        )
        return job.to_dict()
