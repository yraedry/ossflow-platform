"""Estimación de duración de pipelines.

Migrado de ``api/pipeline.py`` en T_LATE_2.5a. Funciones puras + un
único helper que walkea directorios para sumar duraciones via
ffprobe (lazy import de ``api.app.get_video_info`` para evitar el
ciclo).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


def duration_seconds(
    started: Optional[str], completed: Optional[str],
) -> Optional[float]:
    if not started or not completed:
        return None
    try:
        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(completed)
        delta = (t1 - t0).total_seconds()
        return delta if delta > 0 else None
    except (ValueError, TypeError):
        return None


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def total_video_duration(path_str: str) -> Optional[float]:
    """Suma la duración de un fichero o de todos los vídeos en un dir."""
    try:
        from api.app import get_video_info  # lazy: ciclo
    except Exception:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    if p.is_file():
        info = get_video_info(str(p))
        d = info.get("duration") if isinstance(info, dict) else 0
        return float(d) if d else None
    exts = {".mkv", ".mp4", ".avi", ".mov", ".webm"}
    total = 0.0
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            info = get_video_info(str(f))
            d = info.get("duration") if isinstance(info, dict) else 0
            if d:
                total += float(d)
    return total if total > 0 else None
