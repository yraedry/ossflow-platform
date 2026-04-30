"""Servicio de metadata + streaming de vídeos del library.

Encapsula los endpoints ``/api/video-info``, ``/api/thumbnail`` y
``/api/media``. Los tres operan sobre paths del filesystem bajo
``library_path`` o ``MEDIA_ROOT`` y dependen de ``ffmpeg`` / ``ffprobe``.

* ``video_info(path)`` — ffprobe con timeout, devuelve duration, codec,
  resolution, fps. Sin ffprobe (o si falla) → defaults.
* ``generate_thumbnail(path, t)`` — ffmpeg seek+1frame a JPEG en
  memoria. Devuelve bytes o None.
* ``resolve_media_path(path)`` — anti-traversal contra MEDIA_ROOT.

El streaming Range-aware vive en el router porque depende del
``Request`` HTTP — la lógica de parseo se mantiene allí en lugar de
inventarla en una capa intermedia.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_FFPROBE_TIMEOUT = 10
_FFMPEG_TIMEOUT = 15

MEDIA_MIME = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".srt": "application/x-subrip",
    ".vtt": "text/vtt",
}


def _media_root() -> Path:
    return Path(os.environ.get("MEDIA_ROOT", "/media"))


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _parse_fps(rate_str: str) -> float:
    try:
        num, den = rate_str.split("/")
        return round(int(num) / int(den), 2) if int(den) > 0 else 0
    except (ValueError, ZeroDivisionError):
        return 0


def video_info(video_path: str) -> dict[str, Any]:
    """Devuelve metadata de vídeo via ffprobe.

    Si ffprobe falla, devuelve defaults con duration=0. No lanza —
    los consumidores tratan duration=0 como "info no disponible".
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_FFPROBE_TIMEOUT,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            video_stream = next(
                (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
                {},
            )
            duration = float(fmt.get("duration", 0))
            return {
                "duration": duration,
                "duration_formatted": _format_duration(duration),
                "size_mb": round(int(fmt.get("size", 0)) / (1024 * 1024), 1),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "codec": video_stream.get("codec_name", "unknown"),
                "fps": _parse_fps(video_stream.get("r_frame_rate", "0/1")),
            }
    except Exception as exc:  # noqa: BLE001
        log.error("ffprobe failed: %s", exc)
    return {"duration": 0, "duration_formatted": "00:00", "size_mb": 0}


def generate_thumbnail(video_path: str, time_sec: float = 5.0) -> Optional[bytes]:
    """Genera un thumbnail JPEG (320 px ancho) desde un timestamp."""
    try:
        cmd = [
            "ffmpeg", "-ss", str(time_sec), "-i", video_path,
            "-vframes", "1", "-vf", "scale=320:-1",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT)
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except Exception:  # noqa: BLE001
        pass
    return None


def resolve_media_path(path: str) -> Optional[Path]:
    """Resuelve un path bajo ``MEDIA_ROOT`` con anti-traversal.

    Devuelve None si:
    * ``path`` es vacío,
    * resuelve fuera de MEDIA_ROOT,
    * no es un fichero existente.
    """
    if not path:
        return None
    try:
        root = _media_root().resolve()
        target = Path(path).resolve()
        target.relative_to(root)
    except (ValueError, OSError):
        return None
    if not target.is_file():
        return None
    return target
