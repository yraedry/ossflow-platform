"""EasyOCR wrapper with confidence filtering and multi-frame voting."""

import logging
from collections import Counter
from difflib import SequenceMatcher

import cv2
import easyocr
import torch

from ..config import Config
from ..models import OcrResult
from .preprocessor import FramePreprocessor

logger = logging.getLogger(__name__)


class OcrReader:
    """EasyOCR wrapper with confidence filtering and multi-frame voting."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.preprocessor = FramePreprocessor(config)
        try:
            self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
            logger.info("EasyOCR loaded (GPU=%s)", torch.cuda.is_available())
        except RuntimeError:
            self.reader = easyocr.Reader(["en"], gpu=False)
            logger.warning("EasyOCR loaded in CPU mode")

    def _is_valid_text(self, text: str) -> bool:
        """Filter out texts shorter than ocr_min_text_length unless numeric."""
        stripped = text.strip()
        if len(stripped) < self.cfg.ocr_min_text_length:
            # Allow short numeric strings (chapter numbers, etc.)
            if stripped.replace(".", "").replace(",", "").isdigit():
                return True
            return False
        return True

    def _best_variant_read(self, frame) -> OcrResult:
        """Run OCR on all preprocessing variants, return best by confidence."""
        variants = self.preprocessor.variants(frame)
        best = OcrResult(text="", confidence=0.0)

        for img in variants:
            try:
                detections = self.reader.readtext(img, detail=1)
            except Exception:
                continue

            filtered = [
                (txt, conf)
                for (_bbox, txt, conf) in detections
                if conf >= self.cfg.ocr_confidence_min and self._is_valid_text(txt)
            ]
            if not filtered:
                continue

            avg_conf = sum(c for _, c in filtered) / len(filtered)
            combined = " ".join(t for t, _ in filtered).replace("\n", " ").strip()

            if avg_conf > best.confidence and combined:
                best = OcrResult(text=combined, confidence=avg_conf,
                                 raw_texts=[t for t, _ in filtered])

        return best

    def read_frame(self, cap: cv2.VideoCapture, timestamp: float) -> OcrResult:
        """Single-frame OCR at *timestamp* seconds."""
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        if not ok:
            return OcrResult(text="", confidence=0.0)
        return self._best_variant_read(frame)

    def read_voted(self, cap: cv2.VideoCapture, timestamp: float,
                   duration: float) -> OcrResult:
        """Multi-frame voting: sample several frames, cluster, pick by majority.

        Changed from previous behavior (pick highest confidence within largest
        cluster) to pick the text that appears most often (mode / majority vote).
        """
        n = self.cfg.ocr_voting_frames
        window = min(self.cfg.ocr_voting_window, duration - timestamp)
        if window <= 0:
            return self.read_frame(cap, timestamp)

        step = window / max(n - 1, 1)
        readings: list[OcrResult] = []
        for i in range(n):
            t = timestamp + i * step
            r = self.read_frame(cap, t)
            if r.text:
                readings.append(r)

        if not readings:
            return OcrResult(text="", confidence=0.0)

        # Cluster by similarity
        clusters: list[list[OcrResult]] = []
        for r in readings:
            placed = False
            for cluster in clusters:
                ratio = SequenceMatcher(None, cluster[0].text.upper(),
                                        r.text.upper()).ratio()
                if ratio > 0.6:
                    cluster.append(r)
                    placed = True
                    break
            if not placed:
                clusters.append([r])

        # Pick largest cluster (majority vote)
        clusters.sort(key=len, reverse=True)
        best_cluster = clusters[0]

        # Within the majority cluster, pick the most common text (mode)
        text_counter: Counter[str] = Counter()
        text_to_result: dict[str, OcrResult] = {}
        for r in best_cluster:
            normalized = r.text.upper().strip()
            text_counter[normalized] += 1
            # Keep the result with highest confidence for this text
            if (normalized not in text_to_result
                    or r.confidence > text_to_result[normalized].confidence):
                text_to_result[normalized] = r

        most_common_text, _count = text_counter.most_common(1)[0]
        return text_to_result[most_common_text]
