"""Multi-frame stability verification to reject OCR hallucinations.

Supports bidirectional verification: checks frames both before and after
the detection point for more robust confirmation.
"""

from collections import Counter
from difflib import SequenceMatcher
from typing import Optional

import cv2

from ..config import Config
from ..models import OcrResult
from ..ocr.reader import OcrReader


class StabilityVerifier:
    """Multi-frame stability check to reject OCR hallucinations.

    When stability_bidirectional is True, verifies frames both before
    and after the detection point. Returns the most common text (majority
    vote) rather than highest confidence.
    """

    def __init__(self, config: Config, ocr: OcrReader) -> None:
        self.cfg = config
        self.ocr = ocr

    def verify(self, cap: cv2.VideoCapture, timestamp: float,
               reference_text: str, duration: float) -> Optional[OcrResult]:
        """
        Check multiple frames around the detection point. If enough agree
        with the reference text, return the majority-vote reading.
        Otherwise return None.

        When bidirectional is enabled, checks frames before AND after.
        """
        ref_upper = reference_text.upper().strip()
        n = self.cfg.stability_frames
        interval = self.cfg.stability_interval
        agree_threshold = self.cfg.stability_agreement
        bidirectional = self.cfg.stability_bidirectional

        readings: list[OcrResult] = []
        agree_count = 0
        total_checks = 0

        # --- Forward checks (after detection point) ---
        for i in range(1, n + 1):
            t = timestamp + i * interval
            if t >= duration:
                break
            result = self.ocr.read_frame(cap, t)
            future_upper = result.text.upper().strip()
            ratio = SequenceMatcher(None, ref_upper, future_upper).ratio()
            total_checks += 1
            if ratio > 0.6:
                agree_count += 1
                readings.append(result)

        # --- Backward checks (before detection point) ---
        if bidirectional:
            for i in range(1, n + 1):
                t = timestamp - i * interval
                if t < 0:
                    break
                result = self.ocr.read_frame(cap, t)
                past_upper = result.text.upper().strip()
                ratio = SequenceMatcher(None, ref_upper, past_upper).ratio()
                total_checks += 1
                if ratio > 0.6:
                    agree_count += 1
                    readings.append(result)

        if total_checks == 0:
            return None

        agreement_ratio = agree_count / total_checks
        if agreement_ratio >= agree_threshold and readings:
            # Return the majority-vote text (most common), not highest confidence
            return self._majority_vote(readings)

        return None

    @staticmethod
    def _majority_vote(readings: list[OcrResult]) -> OcrResult:
        """Pick the most common text among readings (mode / majority vote).

        Within the most common text group, return the result with highest
        confidence for that text.
        """
        text_counter: Counter[str] = Counter()
        text_to_best: dict[str, OcrResult] = {}

        for r in readings:
            normalized = r.text.upper().strip()
            text_counter[normalized] += 1
            if (normalized not in text_to_best
                    or r.confidence > text_to_best[normalized].confidence):
                text_to_best[normalized] = r

        most_common_text, _count = text_counter.most_common(1)[0]
        return text_to_best[most_common_text]
