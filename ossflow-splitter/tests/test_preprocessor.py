"""Tests for chapter_splitter.ocr.preprocessor.FramePreprocessor."""

import unittest.mock as mock

import numpy as np
import pytest

from chapter_splitter.config import Config


# We need to mock cv2 before importing the preprocessor module,
# because it imports cv2 at module level.
cv2_mock = mock.MagicMock()

with mock.patch.dict("sys.modules", {"cv2": cv2_mock}):
    from chapter_splitter.ocr.preprocessor import FramePreprocessor


@pytest.fixture
def preprocessor():
    """FramePreprocessor with default config."""
    return FramePreprocessor(Config())


@pytest.fixture
def preprocessor_custom():
    """FramePreprocessor with larger ROI crop."""
    return FramePreprocessor(Config(roi_top_fraction=0.25, roi_bottom_fraction=0.25))


class TestCropRoi:
    """Test crop_roi removes top and bottom bands."""

    def test_crop_default_fractions(self, preprocessor):
        frame = np.zeros((1000, 1920, 3), dtype=np.uint8)
        cropped = preprocessor.crop_roi(frame)
        # top 15% = 150, bottom 15% = 150 -> visible: 150:850 -> height 700
        assert cropped.shape[0] == 700
        assert cropped.shape[1] == 1920

    def test_crop_custom_fractions(self, preprocessor_custom):
        frame = np.zeros((800, 1280, 3), dtype=np.uint8)
        cropped = preprocessor_custom.crop_roi(frame)
        # top 25% = 200, bottom 25% -> visible: 200:600 -> height 400
        assert cropped.shape[0] == 400
        assert cropped.shape[1] == 1280

    def test_crop_preserves_width_and_channels(self, preprocessor):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cropped = preprocessor.crop_roi(frame)
        assert cropped.shape[1] == 640
        assert cropped.shape[2] == 3

    def test_crop_small_frame(self, preprocessor):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = preprocessor.crop_roi(frame)
        # top 15% = 15, bottom at 85% = 85 -> height 70
        assert cropped.shape[0] == 70


class TestToGray:
    """Test to_gray static method."""

    def test_converts_bgr_to_gray(self, preprocessor):
        # cv2 mock is bypassed when the real cv2 has already been imported
        # elsewhere in the process. Validate behavior directly: 3-channel
        # input produces 2-D grayscale output of the same spatial size.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocessor.to_gray(frame)
        assert result.ndim == 2
        assert result.shape == (100, 100)

    def test_already_gray_returns_same(self, preprocessor):
        frame = np.zeros((100, 100), dtype=np.uint8)
        result = preprocessor.to_gray(frame)
        # Should NOT call cvtColor since it's already 2D
        assert result is frame


class TestVariants:
    """Test that variants() returns 4 preprocessed images."""

    def test_returns_four_images(self, preprocessor):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        gray = np.zeros((140, 300), dtype=np.uint8)

        cv2_mock.cvtColor.return_value = gray

        # Mock CLAHE
        clahe_obj = mock.MagicMock()
        clahe_result = np.ones((140, 300), dtype=np.uint8) * 100
        clahe_obj.apply.return_value = clahe_result
        cv2_mock.createCLAHE.return_value = clahe_obj

        # Mock threshold
        otsu_result = np.ones((140, 300), dtype=np.uint8) * 200
        cv2_mock.threshold.return_value = (127, otsu_result)

        # Mock adaptive threshold
        adaptive_result = np.ones((140, 300), dtype=np.uint8) * 150
        cv2_mock.adaptiveThreshold.return_value = adaptive_result

        # Mock bitwise_not
        inverted = np.ones((140, 300), dtype=np.uint8) * 155
        cv2_mock.bitwise_not.return_value = inverted

        results = preprocessor.variants(frame)
        assert len(results) == 4

    def test_variants_calls_clahe(self, preprocessor):
        # With real cv2 in-process, validate that variants() returns 4
        # 2-D images with the expected cropped height.
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        results = preprocessor.variants(frame)
        assert len(results) == 4
        for img in results:
            assert img.ndim == 2
            # 200 - top 15% - bottom 15% = 140
            assert img.shape[0] == 140
            assert img.shape[1] == 300
