"""Phrase synchronization with gap-aware time allocation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..config import DubbingConfig

logger = logging.getLogger(__name__)


@dataclass
class SrtBlock:
    """Parsed SRT subtitle block."""
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class PlannedBlock:
    """A block with its allocated time budget.

    ``artificial_gap_to_next_ms`` is populated when a words-level index
    tells us the speaker kept talking through the SRT gap — i.e. the
    gap is a VAD artefact, not a real pause. The pipeline uses this to
    close the silence by shifting the next TTS phrase earlier.
    """

    text: str
    target_start_ms: int
    target_end_ms: int
    allocated_ms: int
    artificial_gap_to_next_ms: int = 0


class SyncAligner:
    """Plan timing for TTS phrases respecting SRT timestamps.

    Each phrase gets its subtitle slot PLUS a share of the gap before the next
    subtitle (when the gap is > inter_phrase_pad_ms). This way Spanish text
    (usually longer) has room without aggressive time compression.
    """

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config

    def plan(
        self,
        blocks: list[SrtBlock],
        video_duration_ms: int | None = None,
        words_index: Optional["WordsIndex"] = None,  # type: ignore[name-defined]
    ) -> list[PlannedBlock]:
        """Return PlannedBlocks with slot + borrowed gap time.

        If ``video_duration_ms`` is given, the last phrase borrows the gap up
        to the end of the video (minus a small tail pad), so the final ES line
        has room without spilling past the video end.

        When ``words_index`` is provided, we also detect **artificial** SRT
        gaps (speaker was still talking but the VAD cut between blocks)
        and record them on each PlannedBlock so the pipeline can close
        those gaps without touching real pauses.
        """
        if not blocks:
            return []

        planned: list[PlannedBlock] = []
        n = len(blocks)
        pad = self.cfg.inter_phrase_pad_ms

        for i, block in enumerate(blocks):
            # Borrow from gap to next subtitle (keep small pad as silence)
            artificial = 0
            if i < n - 1:
                gap_ms = blocks[i + 1].start_ms - block.end_ms
                borrow_ms = max(0, gap_ms - pad)
                if words_index is not None and gap_ms > pad:
                    artificial = _artificial_gap_ms(
                        words_index, block.end_ms, blocks[i + 1].start_ms, pad,
                    )
            else:
                # Last block: borrow whatever remains until the end of the
                # video so a long ES line has room instead of spilling over.
                if video_duration_ms is not None:
                    tail_gap = video_duration_ms - block.end_ms
                    borrow_ms = max(0, tail_gap - pad)
                else:
                    borrow_ms = 0

            slot_ms = max(
                block.duration_ms + borrow_ms,
                self.cfg.min_phrase_duration_ms,
            )

            planned.append(PlannedBlock(
                text=block.text,
                target_start_ms=block.start_ms,
                target_end_ms=block.start_ms + slot_ms,
                allocated_ms=slot_ms,
                artificial_gap_to_next_ms=artificial,
            ))

        return planned


def _artificial_gap_ms(
    words_index,
    gap_start_ms: int,
    gap_end_ms: int,
    pad_ms: int,
) -> int:
    """Return how many ms of an SRT gap are covered by actual speech.

    A gap is "artificial" when the WhisperX words file has words inside
    it — i.e. the SRT block boundaries don't match real speech
    boundaries, typically because the VAD cut through a breath or a
    short disfluency.

    We return the span from the SRT's ``end_ms`` up to the last word
    that still ends before ``gap_end_ms - pad_ms`` (keep the minimum
    breathing pad between phrases). Values ≤0 mean "no artificial
    coverage" and are clamped to 0.
    """
    if gap_end_ms - gap_start_ms <= pad_ms:
        return 0
    # Look for speech that runs into the gap. We trim back by pad_ms so
    # the next phrase still has its micro-silence to start on.
    window_end = gap_end_ms - pad_ms
    last_end = words_index.last_speech_end_within(gap_start_ms, window_end)
    if last_end is None:
        return 0
    return max(0, last_end - gap_start_ms)
