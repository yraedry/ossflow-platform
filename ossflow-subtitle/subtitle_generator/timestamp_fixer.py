"""Timestamp correction for word-level and subtitle-level timing."""

from __future__ import annotations

import logging
from typing import Optional

from .config import SubtitleConfig

log = logging.getLogger("subtitler")


class TimestampFixer:
    """Fix null timestamps, overlaps, gaps, and enforce duration limits."""

    def __init__(self, config: SubtitleConfig) -> None:
        self.config = config

    def fix_words(self, words: list[dict]) -> list[dict]:
        """Fix word-level timestamps: interpolate nulls, remove overlaps."""
        if not words:
            return words
        words = self._interpolate_null_timestamps(words)
        words = self._remove_word_overlaps(words)
        return words

    def fix_subtitles(self, subtitles: list[dict]) -> list[dict]:
        """Fix subtitle-level timing: overlaps, gaps, duration clamping."""
        if not subtitles:
            return subtitles
        subtitles = self._remove_overlaps(subtitles)
        subtitles = self._fill_small_gaps(subtitles)
        subtitles = self._clamp_durations(subtitles)
        return subtitles

    def _interpolate_null_timestamps(self, words: list[dict]) -> list[dict]:
        """Interpolate null start/end times using nearest valid neighbors and character position."""
        # Build list of indices with valid timestamps
        valid_starts: list[tuple[int, float]] = []
        valid_ends: list[tuple[int, float]] = []

        for i, w in enumerate(words):
            if w.get("start") is not None:
                valid_starts.append((i, w["start"]))
            if w.get("end") is not None:
                valid_ends.append((i, w["end"]))

        # Compute cumulative character lengths for proportional interpolation
        char_lengths = [len(w.get("word", "").strip()) or 1 for w in words]
        cum_chars = [0.0]
        for cl in char_lengths:
            cum_chars.append(cum_chars[-1] + cl)

        for i, w in enumerate(words):
            if w.get("start") is None:
                w["start"] = self._interpolate_at(i, valid_starts, valid_ends, cum_chars, is_start=True)
            if w.get("end") is None:
                w["end"] = self._interpolate_at(i, valid_starts, valid_ends, cum_chars, is_start=False)
            # Ensure end >= start
            if w["end"] < w["start"]:
                w["end"] = w["start"] + 0.05

        return words

    def _interpolate_at(
        self,
        idx: int,
        valid_starts: list[tuple[int, float]],
        valid_ends: list[tuple[int, float]],
        cum_chars: list[float],
        is_start: bool,
    ) -> float:
        """Find nearest valid timestamps before/after idx and interpolate by character position."""
        # Find the nearest valid time before this index
        before_time: Optional[float] = None
        before_idx: Optional[int] = None
        for vi, vt in reversed(valid_starts + valid_ends):
            if vi < idx or (vi == idx and not is_start):
                before_time = vt
                before_idx = vi
                break

        # Find the nearest valid time after this index
        after_time: Optional[float] = None
        after_idx: Optional[int] = None
        for vi, vt in valid_starts + valid_ends:
            if vi > idx or (vi == idx and is_start):
                after_time = vt
                after_idx = vi
                break

        if before_time is not None and after_time is not None and before_idx is not None and after_idx is not None:
            # Proportional interpolation by character position
            total_chars = cum_chars[after_idx + 1] - cum_chars[before_idx]
            if total_chars > 0:
                pos_chars = cum_chars[idx] - cum_chars[before_idx]
                if not is_start:
                    pos_chars = cum_chars[idx + 1] - cum_chars[before_idx]
                fraction = pos_chars / total_chars
            else:
                fraction = 0.5
            return before_time + fraction * (after_time - before_time)
        elif before_time is not None:
            return before_time + 0.05
        elif after_time is not None:
            return after_time - 0.05
        else:
            return 0.0

    def _remove_word_overlaps(self, words: list[dict]) -> list[dict]:
        """Ensure word timestamps don't overlap."""
        for i in range(len(words) - 1):
            if words[i]["end"] > words[i + 1]["start"]:
                mid = (words[i]["end"] + words[i + 1]["start"]) / 2
                words[i]["end"] = mid
                words[i + 1]["start"] = mid
        return words

    def _remove_overlaps(self, subtitles: list[dict]) -> list[dict]:
        """If subtitle N overlaps N+1, split the difference."""
        for i in range(len(subtitles) - 1):
            if subtitles[i]["end"] > subtitles[i + 1]["start"]:
                mid = (subtitles[i]["end"] + subtitles[i + 1]["start"]) / 2
                subtitles[i]["end"] = mid
                subtitles[i + 1]["start"] = mid
        return subtitles

    def _fill_small_gaps(self, subtitles: list[dict]) -> list[dict]:
        """Fill gaps smaller than threshold by extending the previous subtitle."""
        threshold = self.config.gap_fill_threshold
        for i in range(len(subtitles) - 1):
            gap = subtitles[i + 1]["start"] - subtitles[i]["end"]
            if 0 < gap < threshold:
                subtitles[i]["end"] = subtitles[i + 1]["start"]
        return subtitles

    def _clamp_durations(self, subtitles: list[dict]) -> list[dict]:
        """Clamp each subtitle duration to [min_duration, max_duration]."""
        min_d = self.config.min_duration
        max_d = self.config.max_duration
        for sub in subtitles:
            duration = sub["end"] - sub["start"]
            if duration < min_d:
                sub["end"] = sub["start"] + min_d
            elif duration > max_d:
                sub["end"] = sub["start"] + max_d
        return subtitles
