"""Lightweight loader for ``<video>.words.json`` sidecars.

Produced by ``subtitle-generator`` after WhisperX alignment. Used here
to distinguish *real* speaker pauses from *artificial* gaps introduced
by the VAD / SRT writer — when we have words inside an SRT gap, the
speaker was still talking and the dubbing pipeline is free to close
that gap instead of inheriting a silence that doesn't exist in the
original audio.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WordsIndex:
    """In-memory index of WhisperX word timings (milliseconds)."""

    def __init__(self, words_ms: list[tuple[int, int]]) -> None:
        self._words = words_ms  # list of (start_ms, end_ms), sorted

    @classmethod
    def load(cls, video_path: Path) -> Optional["WordsIndex"]:
        """Try to load ``<stem>.words.json`` next to ``video_path``."""
        sidecar = video_path.with_name(f"{video_path.stem}.words.json")
        if not sidecar.exists():
            return None
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Invalid words.json: %s", sidecar)
            return None
        if not isinstance(raw, list):
            return None
        words: list[tuple[int, int]] = []
        for w in raw:
            if not isinstance(w, dict):
                continue
            s = w.get("start")
            e = w.get("end")
            if not isinstance(s, (int, float)) or not isinstance(e, (int, float)):
                continue
            start_ms = int(float(s) * 1000)
            end_ms = int(float(e) * 1000)
            if end_ms <= start_ms:
                continue
            words.append((start_ms, end_ms))
        words.sort()
        logger.info("Loaded words index: %d words from %s", len(words), sidecar.name)
        return cls(words)

    def has_speech_in(self, start_ms: int, end_ms: int) -> bool:
        """Return True if any word overlaps the ``[start_ms, end_ms)`` window.

        Binary search would be faster but we do this at most once per SRT
        block (so <200 calls per 40-min video). Linear is fine.
        """
        if end_ms <= start_ms:
            return False
        for w_start, w_end in self._words:
            if w_start >= end_ms:
                return False  # sorted: nothing left can overlap
            if w_end > start_ms:
                return True
        return False

    def last_speech_end_within(
        self, start_ms: int, end_ms: int,
    ) -> Optional[int]:
        """End time of the latest word overlapping the window, or None."""
        last: Optional[int] = None
        for w_start, w_end in self._words:
            if w_start >= end_ms:
                break
            if w_end > start_ms:
                last = max(last or 0, min(w_end, end_ms))
        return last

    def first_speech_start_after(self, t_ms: int) -> Optional[int]:
        """Start time of the first word that begins at or after ``t_ms``."""
        for w_start, _ in self._words:
            if w_start >= t_ms:
                return w_start
        return None

    def speech_coverage_ms(self, start_ms: int, end_ms: int) -> int:
        """Return total ms of speech (words) contained in the window.

        Sums the overlap of each word with ``[start_ms, end_ms)``. Used
        by the pipeline to decide whether a silence gap is synthetic
        (speaker was still talking → close it) or a real pause (speaker
        paused → keep it to preserve lip sync).
        """
        if end_ms <= start_ms:
            return 0
        total = 0
        for w_start, w_end in self._words:
            if w_start >= end_ms:
                break
            overlap_start = max(w_start, start_ms)
            overlap_end = min(w_end, end_ms)
            if overlap_end > overlap_start:
                total += overlap_end - overlap_start
        return total
