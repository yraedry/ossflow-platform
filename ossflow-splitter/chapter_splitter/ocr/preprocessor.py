"""Frame preprocessing for OCR with multiple image variants."""

import cv2
import numpy as np

from ..config import Config


class FramePreprocessor:
    """Produces multiple preprocessed variants of a frame for OCR."""

    def __init__(self, config: Config) -> None:
        self.cfg = config

    def crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """Remove top and bottom bands (logos, watermarks)."""
        h = frame.shape[0]
        top = int(h * self.cfg.roi_top_fraction)
        bottom = int(h * (1.0 - self.cfg.roi_bottom_fraction))
        return frame[top:bottom]

    @staticmethod
    def apply_watermark_mask(
        frame: np.ndarray,
        region: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Paint the given fractional region (x0,y0,x1,y1 in 0..1) black.

        Used to obliterate permanent watermarks (e.g. BJJFanatics logo in the
        bottom-left corner) before running OCR, so they do not bleed into
        detected titles.

        Returns a new array -- input is not mutated.
        """
        if frame is None or frame.size == 0:
            return frame
        x0f, y0f, x1f, y1f = region
        # Clamp to [0, 1]
        x0f = max(0.0, min(1.0, x0f))
        y0f = max(0.0, min(1.0, y0f))
        x1f = max(0.0, min(1.0, x1f))
        y1f = max(0.0, min(1.0, y1f))
        if x1f <= x0f or y1f <= y0f:
            return frame

        h, w = frame.shape[:2]
        x0 = int(w * x0f)
        x1 = int(w * x1f)
        y0 = int(h * y0f)
        y1 = int(h * y1f)

        out = frame.copy()
        out[y0:y1, x0:x1] = 0
        return out

    def mask_watermark(self, frame: np.ndarray) -> np.ndarray:
        """Conditional watermark mask based on config."""
        if not self.cfg.watermark_mask_enabled:
            return frame
        return self.apply_watermark_mask(frame, self.cfg.watermark_region)

    @staticmethod
    def to_gray(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def variants(self, frame: np.ndarray) -> list[np.ndarray]:
        """Return 4 preprocessing variants of a grayscale frame."""
        masked = self.mask_watermark(frame)
        gray = self.to_gray(self.crop_roi(masked))
        results: list[np.ndarray] = []

        # 1. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        results.append(clahe.apply(gray))

        # 2. Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(otsu)

        # 3. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        results.append(adaptive)

        # 4. Inverted CLAHE
        results.append(cv2.bitwise_not(results[0]))

        return results
