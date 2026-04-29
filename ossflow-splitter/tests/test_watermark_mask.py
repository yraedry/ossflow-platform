"""Tests for the watermark-masking helper in FramePreprocessor."""

import unittest.mock as mock

import numpy as np
import pytest

from chapter_splitter.config import Config

# cv2 is imported at module level by preprocessor; mock it before import.
cv2_mock = mock.MagicMock()
with mock.patch.dict("sys.modules", {"cv2": cv2_mock}):
    from chapter_splitter.ocr.preprocessor import FramePreprocessor


@pytest.fixture
def preprocessor():
    return FramePreprocessor(Config())


class TestApplyWatermarkMask:
    def test_blanks_default_bottom_left_region(self):
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)
        out = FramePreprocessor.apply_watermark_mask(frame, (0.0, 0.80, 0.25, 1.00))
        # Bottom-left 25% wide x 20% tall is zeroed: rows 80..100, cols 0..50
        assert np.all(out[80:100, 0:50] == 0)
        # Outside the region untouched
        assert np.all(out[0:80, :] == 255)
        assert np.all(out[80:100, 50:] == 255)

    def test_does_not_mutate_input(self):
        frame = np.full((50, 50, 3), 200, dtype=np.uint8)
        FramePreprocessor.apply_watermark_mask(frame, (0.0, 0.5, 1.0, 1.0))
        assert np.all(frame == 200)

    def test_invalid_region_returns_frame_unchanged(self):
        frame = np.full((20, 20, 3), 123, dtype=np.uint8)
        out = FramePreprocessor.apply_watermark_mask(frame, (0.5, 0.5, 0.3, 0.3))
        assert np.all(out == 123)

    def test_clamps_out_of_range_fractions(self):
        frame = np.full((40, 40, 3), 50, dtype=np.uint8)
        out = FramePreprocessor.apply_watermark_mask(frame, (-0.5, -0.5, 2.0, 2.0))
        # The entire frame gets blanked because region is clamped to (0,0,1,1).
        assert np.all(out == 0)

    def test_empty_frame_returns_empty(self):
        frame = np.zeros((0, 0, 3), dtype=np.uint8)
        out = FramePreprocessor.apply_watermark_mask(frame, (0.0, 0.0, 1.0, 1.0))
        assert out.size == 0


class TestMaskWatermarkConditional:
    def test_disabled_returns_frame_unchanged(self):
        cfg = Config(watermark_mask_enabled=False)
        pre = FramePreprocessor(cfg)
        frame = np.full((50, 50, 3), 200, dtype=np.uint8)
        out = pre.mask_watermark(frame)
        assert out is frame

    def test_enabled_applies_mask(self, preprocessor):
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)
        out = preprocessor.mask_watermark(frame)
        # Default region: x [0, 50), y [80, 100) zeroed
        assert np.all(out[80:100, 0:50] == 0)
        assert np.any(out == 255)
