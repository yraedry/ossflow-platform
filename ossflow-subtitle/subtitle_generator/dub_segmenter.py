"""Segment speech from word-level timestamps for dubbing (nivel 3).

Input: list of WhisperX word-timestamps (from <video>.words.json).
Output: list of speech segments grouped by real speaker pauses.

Unlike the reading-oriented SRT, segments here mirror how the speaker
actually breathes — so a TTS rendered per segment respires with the video
instead of inheriting the gaps/long slots of the subtitle track.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SegmenterConfig:
    """Thresholds for grouping words into speech segments."""

    # Pause >= this opens a new segment. 450 ms = mid-ground between
    # fluency and sync: each TTS render carries more words, so prosody
    # arcs across them instead of restarting every 2-3 words. 250 ms
    # over-fragmented for XTTS (61 segments on a 158-s clip → 46 hard
    # cuts because every boundary restarts timbre). 450 typically
    # lands around ~35 segments, with arcs long enough to sound like
    # continuous speech without missing real coach pauses.
    min_pause_ms: int = 450
    # Hard cap per segment. Chatterbox char_limit is 260; 220 keeps headroom
    # for the ES adaptation (usually 10-20% longer than EN).
    max_chars: int = 220
    # If a segment would exceed max_duration_s, force-split on the next word.
    # Protects against speakers without pauses.
    max_duration_s: float = 12.0
    # Drop micro-segments shorter than this — usually alignment artifacts.
    min_duration_s: float = 0.3


def load_words(words_path: Path) -> list[dict]:
    """Load <video>.words.json into a list of word dicts."""
    raw = json.loads(Path(words_path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"words.json must be a list, got {type(raw).__name__}")
    return raw


def segment_speech(
    words: list[dict],
    config: SegmenterConfig | None = None,
) -> list[dict]:
    """Group ``words`` into speech segments separated by real pauses.

    Each returned segment has ``start``, ``end`` (seconds), ``text``,
    ``duration_ms``, and the raw ``words`` it contains (for downstream
    re-alignment if the translator changes word count).
    """
    cfg = config or SegmenterConfig()
    if not words:
        return []

    clean = [w for w in words if str(w.get("word", "")).strip()]
    clean.sort(key=lambda w: float(w.get("start", 0.0)))

    segments: list[dict] = []
    buf: list[dict] = []
    buf_chars = 0

    def _flush() -> None:
        if not buf:
            return
        start = float(buf[0]["start"])
        end = float(buf[-1]["end"])
        dur = end - start
        if dur < cfg.min_duration_s:
            buf.clear()
            return
        text = " ".join(str(w["word"]).strip() for w in buf)
        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "duration_ms": int(round(dur * 1000)),
            "words": list(buf),
        })
        buf.clear()

    prev_end: float | None = None
    for w in clean:
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", w_start))
        w_text = str(w.get("word", "")).strip()
        if not w_text:
            continue

        addition = len(w_text) + (1 if buf else 0)
        gap_ms = (w_start - prev_end) * 1000 if prev_end is not None else 0.0
        cur_dur = (w_end - float(buf[0]["start"])) if buf else 0.0

        split = False
        if buf and gap_ms >= cfg.min_pause_ms:
            split = True
        elif buf and buf_chars + addition > cfg.max_chars:
            split = True
        elif buf and cur_dur > cfg.max_duration_s:
            split = True

        if split:
            _flush()
            buf_chars = 0

        buf.append(w)
        buf_chars += len(w_text) + (1 if buf_chars else 0)
        prev_end = w_end

    _flush()
    return segments


def segments_to_srt_blocks(segments: list[dict]) -> list[dict]:
    """Convert speech segments to SRT-style blocks (start/end in seconds).

    Text comes in unchanged — translation is applied separately.
    """
    return [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in segments
    ]
