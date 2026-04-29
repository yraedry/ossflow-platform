"""State-machine chapter detector combining voice analysis and OCR.

Improvements over v1:
- UNCERTAIN state for hysteresis zone (0.20-0.30)
- Per-chapter cooldown (hash of title) instead of global cooldown
- Bidirectional stability verification
- Reduced similarity threshold (0.75) for subtitle transitions
- Background memory learns during both voice and detection phases
"""

import hashlib
import logging
import re
import string
from collections import deque
from difflib import SequenceMatcher

import cv2
import numpy as np

from ..config import Config
from ..models import Chapter
from ..utils import sanitize_filename
from ..ocr.reader import OcrReader
from ..audio.analyzer import AudioAnalyzer
from .background_memory import BackgroundMemory
from .stability import StabilityVerifier

logger = logging.getLogger(__name__)

# Similarity threshold for detecting title transitions
TITLE_SIMILARITY_THRESHOLD = 0.75


class ChapterDetector:
    """State-machine chapter detector combining voice analysis and OCR.

    States:
    - VOICE: voice is present, learn background text
    - SILENT: silence detected, run detection logic
    - UNCERTAIN: hysteresis zone, run OCR with higher threshold
    """

    def __init__(self, config: Config, ocr: OcrReader,
                 audio: AudioAnalyzer) -> None:
        self.cfg = config
        self.ocr = ocr
        self.audio = audio
        self.memory = BackgroundMemory(config)
        self.stability = StabilityVerifier(config, ocr)

    @staticmethod
    def _title_hash(title: str) -> str:
        """Compute a hash for cooldown tracking per chapter title."""
        return hashlib.md5(title.upper().strip().encode()).hexdigest()[:12]

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        """Lowercase and strip punctuation/whitespace for blacklist/dedupe."""
        if not text:
            return ""
        s = text.lower()
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _is_blacklisted(self, text: str) -> bool:
        """True if *text* contains any blacklist token (case-insensitive substr)."""
        norm = self._normalize_for_match(text)
        if not norm:
            return False
        for token in self.cfg.ocr_blacklist:
            tok = self._normalize_for_match(token)
            if tok and tok in norm:
                return True
        return False

    @staticmethod
    def _title_similarity(a: str, b: str) -> float:
        """Levenshtein-like normalized similarity using SequenceMatcher."""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.upper().strip(), b.upper().strip()).ratio()

    def _are_titles_duplicate(self, a: str, b: str) -> bool:
        """True if two titles should be considered the same chapter."""
        if not a or not b:
            return False
        na = self._normalize_for_match(a)
        nb = self._normalize_for_match(b)
        if not na or not nb:
            return False
        if na == nb:
            return True
        # Substring after normalization
        if na in nb or nb in na:
            return True
        ratio = SequenceMatcher(None, na, nb).ratio()
        return (1.0 - ratio) <= self.cfg.dedupe_similarity_threshold

    @staticmethod
    def _compute_hsv_hist(frame: np.ndarray) -> np.ndarray:
        """Compute normalized HSV histogram for color-corroboration."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _color_transition_score(self, cap: cv2.VideoCapture,
                                timestamp: float,
                                history: "deque[np.ndarray]") -> float:
        """Chi-square distance between current frame HSV hist and mean of history.

        Returns 0.0 when no history is available.
        """
        if not self.cfg.color_corroboration or not history:
            return 0.0
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        if not ok:
            return 0.0
        try:
            cur = self._compute_hsv_hist(frame)
        except Exception:
            return 0.0
        mean_hist = np.mean(np.stack(list(history), axis=0), axis=0)
        try:
            dist = float(cv2.compareHist(cur.astype("float32"),
                                         mean_hist.astype("float32"),
                                         cv2.HISTCMP_CHISQR))
        except Exception:
            return 0.0
        return dist

    def _find_scene_cut(self, cap: cv2.VideoCapture, timestamp: float) -> float:
        """Search backward from timestamp to find scene transition via MSE."""
        window = self.cfg.cut_search_window
        start = max(0.0, timestamp - window)
        step = 0.1  # 100ms steps backward

        prev_frame = None
        best_cut = timestamp
        best_mse = 0.0

        t = timestamp
        while t >= start:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                t -= step
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            if prev_frame is not None:
                mse = float(np.mean((gray - prev_frame) ** 2))
                if mse > best_mse and mse > self.cfg.cut_mse_threshold:
                    best_mse = mse
                    best_cut = t

            prev_frame = gray
            t -= step

        return best_cut

    def detect(self, video_path: str, cap: cv2.VideoCapture,
               duration: float) -> list[Chapter]:
        """Scan the video and return detected chapters."""
        chapters: list[Chapter] = []
        current_start = 0.0
        current_title = "Introduction"

        # Per-chapter cooldown tracking: title_hash -> last_detect_time
        cooldown_map: dict[str, float] = {}

        step = self.cfg.scan_step
        t = 0.0
        decay_counter = 0

        # Rolling HSV histogram memory for color-based transition corroboration.
        # Keep ~1 minute of samples at scan_step rate.
        color_history_len = max(1, int(60.0 / max(step, 0.1)))
        color_history: deque[np.ndarray] = deque(maxlen=color_history_len)
        color_sample_counter = 0

        logger.info("Scanning for chapters (step=%.1fs)...", step)

        while t < duration:
            # Progress
            minute = int(t / 60)
            if minute > 0 and abs(t - minute * 60) < step:
                pct = int(t / duration * 100)
                logger.info("  Scan: minute %d (%d%%)", minute, pct)

            second = int(t)

            # Periodic decay of background memory
            decay_counter += 1
            if decay_counter % 120 == 0:  # every ~60 seconds at 0.5s step
                self.memory.decay()

            # Sample HSV histogram for color corroboration (every ~2 steps)
            if self.cfg.color_corroboration:
                color_sample_counter += 1
                if color_sample_counter % 2 == 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ok_c, frame_c = cap.read()
                    if ok_c:
                        try:
                            color_history.append(self._compute_hsv_hist(frame_c))
                        except Exception:
                            pass

            # --- Voice check (pre-computed, with hysteresis) ---
            is_silent = self.audio.is_silent(second)
            is_uncertain = self.audio.is_uncertain(second)

            # --- OCR ---
            result = self.ocr.read_frame(cap, t)
            text_clean = sanitize_filename(result.text).upper()

            # Always learn background (not just during voice)
            if text_clean and len(text_clean) > 3:
                self.memory.learn(text_clean)

            if not is_silent and not is_uncertain:
                # VOICE state: skip detection
                t += step
                continue

            # --- SILENT or UNCERTAIN: detection mode ---

            # For UNCERTAIN state, require higher OCR confidence
            if is_uncertain and result.confidence < self.cfg.ocr_confidence_min + 0.10:
                t += step
                continue

            # Must have meaningful text
            if len(text_clean) <= 3:
                t += step
                continue

            # Blacklist filter (e.g. "BJJFANATICS" watermark bleed)
            if self._is_blacklisted(text_clean):
                logger.debug("Discarding blacklisted OCR text: '%s'", text_clean)
                t += step
                continue

            # Minimum chars for a chapter title (post-strip)
            if len(text_clean.strip()) < self.cfg.ocr_min_title_chars:
                t += step
                continue

            # Skip very long text (disclaimers / legal)
            if len(text_clean) > self.cfg.max_title_length:
                t += step
                continue

            # Check against background memory
            if self.memory.is_background(text_clean):
                t += step
                continue

            # --- Potential chapter title ---

            # Per-chapter cooldown check
            title_h = self._title_hash(text_clean)
            last_detect = cooldown_map.get(title_h, -999.0)
            if (t - last_detect) < self.cfg.cooldown_seconds:
                t += step
                continue

            # Multi-frame voted read for better accuracy
            voted = self.ocr.read_voted(cap, t, duration)
            if voted.text:
                text_clean = sanitize_filename(voted.text).upper()
                result = voted

            # Re-run blacklist on voted text (it may differ)
            if self._is_blacklisted(text_clean):
                logger.debug("Discarding blacklisted voted title: '%s'",
                             text_clean)
                t += step
                continue

            # --- Audio/color corroboration boost ---
            effective_conf = float(result.confidence)
            if self.cfg.audio_corroboration and hasattr(
                    self.audio, "is_transition_window"):
                try:
                    if self.audio.is_transition_window(t):
                        effective_conf += self.cfg.corroboration_boost_audio
                except Exception:
                    pass
            if self.cfg.color_corroboration and color_history:
                color_score = self._color_transition_score(cap, t, color_history)
                # Heuristic: chi-sq > 2.0 typically signals a real color change
                if color_score > 2.0:
                    effective_conf += self.cfg.corroboration_boost_color

            # Title-level confidence gate (higher bar than generic OCR)
            if effective_conf < self.cfg.ocr_title_confidence_min:
                logger.debug(
                    "Rejecting title '%s' -- confidence %.2f < %.2f",
                    text_clean, effective_conf,
                    self.cfg.ocr_title_confidence_min,
                )
                t += step
                continue

            # Skip if same as current title (reduced threshold 0.75)
            if SequenceMatcher(None, current_title.upper(),
                               text_clean).ratio() > TITLE_SIMILARITY_THRESHOLD:
                t += step
                continue

            # Stability verification (bidirectional when enabled)
            stable = self.stability.verify(cap, t, text_clean, duration)
            if stable is None:
                # Hallucination -- did not persist
                t += step
                continue

            # --- Confirmed new chapter ---
            use_text = sanitize_filename(stable.text).upper()
            if not use_text or len(use_text) <= 3:
                use_text = text_clean

            cut_point = self._find_scene_cut(cap, t)
            prev_duration = cut_point - current_start

            min_dur = self.cfg.min_chapter_duration_sec
            if prev_duration < min_dur and chapters:
                # Prev candidate too short -> discard this transition, keep
                # accumulating under the prior chapter title.
                logger.info(
                    "  Discarding short chapter '%s' (%.0fs < %.0fs minimum)",
                    current_title, prev_duration, min_dur,
                )
                t += step
                continue

            if prev_duration > self.cfg.min_chapter_duration or not chapters:
                chapters.append(Chapter(current_start, cut_point, current_title))
                logger.info("  Chapter saved: '%s' (%.0fs)", current_title,
                            prev_duration)

            current_start = cut_point
            current_title = use_text

            # Update per-chapter cooldown
            new_title_h = self._title_hash(use_text)
            cooldown_map[new_title_h] = t

            logger.info("  New chapter: '%s' at %.1fs", current_title, t)

            # Skip ahead past title card
            t += 2.0
            continue

        # Final chapter
        chapters.append(Chapter(current_start, None, current_title))
        logger.info("  Final chapter: '%s'", current_title)

        # --- Post-processing: dedupe consecutive near-identical titles ---
        chapters = self._dedupe_consecutive(chapters)

        # --- Post-processing: drop still-too-short chapters by merging them
        # into the previous one.
        chapters = self._enforce_min_duration(chapters, duration)

        logger.info("Total chapters detected: %d", len(chapters))
        return chapters

    def _dedupe_consecutive(self, chapters: list[Chapter]) -> list[Chapter]:
        """Merge consecutive chapters whose titles are near-duplicates."""
        if len(chapters) < 2:
            return chapters
        out: list[Chapter] = [chapters[0]]
        for nxt in chapters[1:]:
            prev = out[-1]
            if self._are_titles_duplicate(prev.title, nxt.title):
                logger.info(
                    "  Merging duplicate chapters '%s' + '%s'",
                    prev.title, nxt.title,
                )
                out[-1] = Chapter(prev.start, nxt.end, prev.title)
            else:
                out.append(nxt)
        return out

    def _enforce_min_duration(self, chapters: list[Chapter],
                              duration: float) -> list[Chapter]:
        """Drop chapters shorter than min_chapter_duration_sec by merging into prev."""
        if not chapters:
            return chapters
        min_dur = self.cfg.min_chapter_duration_sec
        out: list[Chapter] = []
        for ch in chapters:
            end = ch.end if ch.end is not None else duration
            ch_dur = end - ch.start
            if ch_dur < min_dur and out:
                logger.info(
                    "  Discarding short chapter '%s' (%.0fs < %.0fs minimum)",
                    ch.title, ch_dur, min_dur,
                )
                prev = out[-1]
                out[-1] = Chapter(prev.start, ch.end, prev.title)
            else:
                out.append(ch)
        return out
