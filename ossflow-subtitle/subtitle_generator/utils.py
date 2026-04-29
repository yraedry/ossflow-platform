"""Shared utility functions for the subtitle generator."""

from __future__ import annotations


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    millis = int(round((seconds % 1) * 1000))
    total_secs = int(seconds)
    hrs = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"
