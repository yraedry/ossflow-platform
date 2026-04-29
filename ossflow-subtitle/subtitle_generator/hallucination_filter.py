"""Post-transcription filters to remove or fix hallucinated content."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from .config import SubtitleConfig

log = logging.getLogger("subtitler")


class HallucinationFilter:
    """Post-transcription filters to remove or fix hallucinated content."""

    def __init__(self, config: SubtitleConfig, initial_prompt: str = "") -> None:
        self.config = config
        self._initial_prompt = initial_prompt.strip().lower()
        self._seen_texts: set[str] = set()
        self.stats = {
            "prompt_echo": 0,
            "garbage_text": 0,
            "repeated_segments": 0,
            "repeated_phrases": 0,
            "low_confidence": 0,
            "nonsense_rate": 0,
            "impossible_timing": 0,
            "silence_hallucinations": 0,
            "synthetic_scores": 0,
        }

    def filter_all(
        self,
        segments: list[dict],
        audio_path: Optional[str | Path] = None,
        return_dropped: bool = False,
    ) -> list[dict] | tuple[list[dict], list[dict]]:
        """Apply all filters in sequence and return cleaned segments.

        Parameters
        ----------
        segments:
            Raw segments from WhisperX transcription/alignment.
        audio_path:
            Path to the source audio/video file.  When provided the silence
            filter can analyse actual audio energy; otherwise it is skipped.
        return_dropped:
            When True, return a 2-tuple ``(kept, dropped)`` where *dropped* is
            a list of ``{start, end, text, reason}`` dicts for every removed segment.
        """
        input_ids = {id(s) for s in segments}

        def _track(name: str, before: list[dict], after: list[dict]) -> list[dict]:
            after_ids = {id(s) for s in after}
            for s in before:
                if id(s) not in after_ids:
                    dropped.append({
                        "start": round(s.get("start", 0), 3),
                        "end": round(s.get("end", 0), 3),
                        "text": s.get("text", ""),
                        "reason": name,
                    })
            return after

        dropped: list[dict] = []

        after = self._filter_prompt_echo(segments)
        _track("prompt_echo", segments, after); segments = after

        after = self._filter_garbage_text(segments)
        _track("garbage_text", segments, after); segments = after

        after = self._filter_impossible_timing(segments)
        _track("impossible_timing", segments, after); segments = after

        after = self._filter_repeated_segments(segments)
        _track("repeated_segments", segments, after); segments = after

        after = self._filter_repeated_phrases(segments)
        _track("repeated_phrases", segments, after); segments = after

        after = self._filter_low_confidence(segments)
        _track("low_confidence", segments, after); segments = after

        after = self._filter_nonsense_rate(segments)
        _track("nonsense_rate", segments, after); segments = after

        after = self._filter_silence_hallucinations(segments, audio_path)
        _track("silence_hallucinations", segments, after); segments = after

        after = self._filter_synthetic_scores(segments)
        _track("synthetic_scores", segments, after); segments = after

        total_dropped = sum(self.stats.values())
        if total_dropped > 0:
            log.info(
                "Hallucination filter: dropped/modified %d items "
                "(prompt_echo=%d, garbage=%d, repeated_seg=%d, repeated_phrase=%d, low_conf=%d, "
                "nonsense=%d, bad_timing=%d, silence=%d, synthetic=%d)",
                total_dropped,
                self.stats["prompt_echo"],
                self.stats["garbage_text"],
                self.stats["repeated_segments"],
                self.stats["repeated_phrases"],
                self.stats["low_confidence"],
                self.stats["nonsense_rate"],
                self.stats["impossible_timing"],
                self.stats["silence_hallucinations"],
                self.stats["synthetic_scores"],
            )
        if return_dropped:
            return segments, dropped
        return segments

    def _filter_prompt_echo(self, segments: list[dict]) -> list[dict]:
        """Drop segments that echo back (part of) the initial prompt.

        Whisper sometimes hallucinates the initial prompt as transcription,
        especially during silence or low-energy sections.  We detect this by
        checking if a substantial substring of the segment text appears inside
        the initial prompt.
        """
        if not self._initial_prompt:
            return segments

        result: list[dict] = []
        for seg in segments:
            text = (seg.get("text") or "").strip().lower()
            if not text or len(text) < 15:
                result.append(seg)
                continue

            # Check if the segment text is substantially contained in the prompt
            ratio = SequenceMatcher(None, self._initial_prompt, text).ratio()
            if ratio >= 0.55:
                self.stats["prompt_echo"] += 1
                log.debug("Dropped prompt-echo segment (%.0f%% match): %r",
                          ratio * 100, text[:80])
                continue
            result.append(seg)
        return result

    def _filter_garbage_text(self, segments: list[dict]) -> list[dict]:
        """Drop segments with URLs, repeated tokens, or other non-speech garbage."""
        result: list[dict] = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            t = text.lower()
            # URLs, domains, email-like
            if re.search(r'\.com|\.net|\.org|\.co\.|www\.|http|@\w+\.\w+', t):
                self.stats["garbage_text"] += 1
                log.debug("Dropped garbage (URL/domain): %r", text[:80])
                continue
            # Music/sound markers
            if re.search(r'♪|♫|\[music\]|\[applause\]|\(music\)', t):
                self.stats["garbage_text"] += 1
                log.debug("Dropped garbage (music marker): %r", text[:80])
                continue
            # All-caps short garbage (often OCR of on-screen text)
            words = text.split()
            if len(words) <= 4 and text == text.upper() and len(text) > 5:
                self.stats["garbage_text"] += 1
                log.debug("Dropped garbage (all-caps short): %r", text[:80])
                continue
            # Single word repeated 4+ times — classic Whisper hallucination on silence/noise
            if len(words) >= 4 and len(set(w.lower() for w in words)) == 1:
                self.stats["garbage_text"] += 1
                log.debug("Dropped garbage (single word repeated): %r", text[:80])
                continue
            result.append(seg)
        return result

    def _filter_impossible_timing(self, segments: list[dict]) -> list[dict]:
        """Remove segments with end <= start, negative duration, or out-of-order timestamps."""
        result: list[dict] = []
        prev_end = -1.0
        for seg in segments:
            start = seg.get("start")
            end = seg.get("end")
            if start is None or end is None:
                result.append(seg)
                continue
            if end <= start or start < 0:
                self.stats["impossible_timing"] += 1
                log.debug("Dropped impossible timing: start=%.3f end=%.3f text=%r",
                          start, end, seg.get("text", "")[:60])
                continue
            if start < prev_end - 0.05:
                # Out-of-order segment (more than 50ms overlap with previous)
                self.stats["impossible_timing"] += 1
                log.debug("Dropped out-of-order segment: start=%.3f < prev_end=%.3f",
                          start, prev_end)
                continue
            prev_end = end
            result.append(seg)
        return result

    def _filter_repeated_segments(self, segments: list[dict]) -> list[dict]:
        """Drop segments that are >80% similar to any of the previous N segments.

        Also maintains a global set of unique normalised texts so that exact
        duplicates are caught regardless of distance (lookback-independent).
        """
        result: list[dict] = []
        lookback = self.config.similarity_lookback
        threshold = self.config.similarity_threshold

        for seg in segments:
            text = seg.get("text", "").strip().lower()
            if not text:
                continue

            # Global exact-duplicate check (distance-independent)
            if text in self._seen_texts:
                self.stats["repeated_segments"] += 1
                log.debug("Dropped global duplicate segment: %r", text[:80])
                continue

            is_repeat = False
            # Compare against the last N accepted segments
            for prev in result[-lookback:]:
                prev_text = prev.get("text", "").strip().lower()
                if not prev_text:
                    continue
                ratio = SequenceMatcher(None, prev_text, text).ratio()
                if ratio >= threshold:
                    is_repeat = True
                    break
            if is_repeat:
                self.stats["repeated_segments"] += 1
                log.debug("Dropped repeated segment: %r", text[:80])
            else:
                self._seen_texts.add(text)
                result.append(seg)
        return result

    def _filter_repeated_phrases(self, segments: list[dict]) -> list[dict]:
        """Truncate segments where a 3-word ngram repeats more than twice."""
        ngram_size = self.config.repeated_ngram_size
        max_count = self.config.repeated_ngram_max

        for seg in segments:
            text = seg.get("text", "").strip()
            words = text.split()
            if len(words) < ngram_size * (max_count + 1):
                continue

            ngram_positions: dict[str, list[int]] = {}
            for i in range(len(words) - ngram_size + 1):
                gram = " ".join(words[i:i + ngram_size]).lower()
                ngram_positions.setdefault(gram, []).append(i)

            for gram, positions in ngram_positions.items():
                if len(positions) > max_count:
                    # Truncate to just after the first occurrence
                    cut_at = positions[0] + ngram_size
                    truncated = " ".join(words[:cut_at])
                    seg["text"] = truncated
                    # Also truncate word-level data if present
                    if "words" in seg:
                        seg["words"] = seg["words"][:cut_at]
                    self.stats["repeated_phrases"] += 1
                    log.debug("Truncated repeated phrase '%s' in segment", gram)
                    break
        return segments

    def _filter_low_confidence(self, segments: list[dict]) -> list[dict]:
        """Drop segments where >60% of aligned words have confidence below threshold."""
        result: list[dict] = []
        word_threshold = self.config.low_confidence_word_threshold
        seg_ratio = self.config.low_confidence_segment_ratio

        for seg in segments:
            words = seg.get("words", [])
            if not words:
                result.append(seg)
                continue

            scored_words = [w for w in words if "score" in w and w["score"] is not None]
            if not scored_words:
                result.append(seg)
                continue

            low_count = sum(1 for w in scored_words if w["score"] < word_threshold)
            ratio = low_count / len(scored_words)

            if ratio >= seg_ratio:
                self.stats["low_confidence"] += 1
                log.debug("Dropped low-confidence segment (%.0f%% below %.1f): %r",
                          ratio * 100, word_threshold, seg.get("text", "")[:60])
            else:
                result.append(seg)
        return result

    def _filter_nonsense_rate(self, segments: list[dict]) -> list[dict]:
        """Drop segments with unrealistic characters-per-second rate."""
        result: list[dict] = []
        max_cps = self.config.max_chars_per_second

        for seg in segments:
            start = seg.get("start")
            end = seg.get("end")
            text = seg.get("text", "").strip()
            if start is None or end is None or not text:
                result.append(seg)
                continue
            duration = end - start
            if duration <= 0:
                result.append(seg)
                continue
            cps = len(text) / duration
            if cps > max_cps:
                self.stats["nonsense_rate"] += 1
                log.debug("Dropped nonsense rate (%.1f chars/s): %r", cps, text[:60])
            else:
                result.append(seg)
        return result

    # ------------------------------------------------------------------
    # Filter 6: Silence hallucinations
    # ------------------------------------------------------------------

    def _filter_silence_hallucinations(
        self,
        segments: list[dict],
        audio_path: Optional[str | Path] = None,
    ) -> list[dict]:
        """Remove segments whose audio range is effectively silence.

        Uses *pydub* to load the audio and measure RMS in the time range
        ``[start, end]`` of each segment.  If the RMS is below the configured
        silence threshold (``silence_threshold_db``, default -40 dBFS) the
        segment is considered a hallucination over silence and is dropped.

        When *audio_path* is ``None`` the filter is a no-op (graceful skip).
        """
        if audio_path is None:
            return segments

        try:
            from pydub import AudioSegment as PydubSegment
        except ImportError:
            log.warning("pydub not installed -- skipping silence filter")
            return segments

        audio_path = Path(audio_path)
        if not audio_path.exists():
            log.warning("Audio file not found for silence filter: %s", audio_path)
            return segments

        try:
            audio = PydubSegment.from_file(str(audio_path))
        except Exception as exc:
            log.warning("Could not load audio for silence filter: %s", exc)
            return segments

        threshold_db = self.config.silence_threshold_db
        result: list[dict] = []

        for seg in segments:
            start = seg.get("start")
            end = seg.get("end")
            if start is None or end is None:
                result.append(seg)
                continue

            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            chunk = audio[start_ms:end_ms]

            if len(chunk) == 0:
                result.append(seg)
                continue

            if chunk.dBFS < threshold_db:
                self.stats["silence_hallucinations"] += 1
                log.debug(
                    "Dropped silence hallucination (%.1f dBFS): %r",
                    chunk.dBFS,
                    seg.get("text", "")[:60],
                )
            else:
                result.append(seg)

        return result

    # ------------------------------------------------------------------
    # Filter 7: Synthetic score detection
    # ------------------------------------------------------------------

    def _filter_synthetic_scores(self, segments: list[dict]) -> list[dict]:
        """Apply stricter confidence filtering to segments with synthetic scores.

        When alignment fails, ``_synthetic_word_timing`` assigns a score of
        exactly ``0.5`` to every word.  This filter detects segments where
        more than ``synthetic_score_ratio`` (default 50%) of scored words
        have a score of exactly 0.5 and re-evaluates them with a much
        stricter ``low_confidence_segment_ratio`` (``synthetic_score_strict_ratio``,
        default 0.30).
        """
        synthetic_ratio_threshold = self.config.synthetic_score_ratio
        strict_seg_ratio = self.config.synthetic_score_strict_ratio
        word_threshold = self.config.low_confidence_word_threshold

        result: list[dict] = []
        for seg in segments:
            words = seg.get("words", [])
            if not words:
                result.append(seg)
                continue

            scored = [w for w in words if "score" in w and w["score"] is not None]
            if not scored:
                result.append(seg)
                continue

            exact_half = sum(1 for w in scored if w["score"] == 0.5)
            ratio_synthetic = exact_half / len(scored)

            if ratio_synthetic > synthetic_ratio_threshold:
                # This looks like a synthetic-timed segment -- apply strict filter
                low_count = sum(1 for w in scored if w["score"] < word_threshold)
                low_ratio = low_count / len(scored)
                if low_ratio >= strict_seg_ratio:
                    self.stats["synthetic_scores"] += 1
                    log.debug(
                        "Dropped synthetic-scored segment (%.0f%% exact-0.5, "
                        "%.0f%% low-conf): %r",
                        ratio_synthetic * 100,
                        low_ratio * 100,
                        seg.get("text", "")[:60],
                    )
                    continue
            result.append(seg)
        return result
