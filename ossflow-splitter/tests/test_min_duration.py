"""Tests for min-duration enforcement in ChapterDetector._enforce_min_duration."""

import unittest.mock as mock

import pytest

from chapter_splitter.config import Config
from chapter_splitter.models import Chapter

cv2_mock = mock.MagicMock()
with mock.patch.dict("sys.modules", {"cv2": cv2_mock}):
    from chapter_splitter.detection.detector import ChapterDetector


def _make_detector(cfg: Config) -> ChapterDetector:
    det = object.__new__(ChapterDetector)
    det.cfg = cfg
    return det


class TestEnforceMinDuration:
    def test_drops_short_chapter_and_extends_previous(self):
        det = _make_detector(Config(min_chapter_duration_sec=60.0))
        chapters = [
            Chapter(0.0, 120.0, "Intro"),
            Chapter(120.0, 162.0, "Too Short"),  # 42s < 60s
            Chapter(162.0, 300.0, "Next"),
        ]
        out = det._enforce_min_duration(chapters, duration=300.0)
        assert len(out) == 2
        assert out[0].title == "Intro"
        assert out[0].start == 0.0
        assert out[0].end == 162.0
        assert out[1].title == "Next"

    def test_keeps_chapter_meeting_minimum(self):
        det = _make_detector(Config(min_chapter_duration_sec=60.0))
        chapters = [
            Chapter(0.0, 120.0, "A"),
            Chapter(120.0, 240.0, "B"),  # 120s
        ]
        out = det._enforce_min_duration(chapters, duration=240.0)
        assert len(out) == 2

    def test_handles_open_ended_final_chapter(self):
        det = _make_detector(Config(min_chapter_duration_sec=60.0))
        chapters = [
            Chapter(0.0, 200.0, "Main"),
            Chapter(200.0, None, "Outro"),  # duration resolves via param
        ]
        # Outro runs from 200..210 -> 10s, should be dropped.
        out = det._enforce_min_duration(chapters, duration=210.0)
        assert len(out) == 1
        assert out[0].title == "Main"
        assert out[0].end is None or out[0].end == 210.0

    def test_first_chapter_never_dropped_even_if_short(self):
        det = _make_detector(Config(min_chapter_duration_sec=60.0))
        chapters = [Chapter(0.0, 10.0, "Short Intro")]
        out = det._enforce_min_duration(chapters, duration=10.0)
        assert len(out) == 1

    def test_zero_chapters(self):
        det = _make_detector(Config())
        out = det._enforce_min_duration([], duration=100.0)
        assert out == []

    def test_default_min_is_60_seconds(self):
        assert Config().min_chapter_duration_sec == 60.0


class TestColorHistHelper:
    def test_compute_hsv_hist_returns_normalized_array(self):
        """_compute_hsv_hist should return a 2-D normalized histogram."""
        import numpy as np
        try:
            import cv2  # noqa: F401
        except Exception:
            pytest.skip("cv2 not available")

        det = _make_detector(Config())
        # Non-trivial frame so cv2.calcHist returns something meaningful.
        frame = (np.random.rand(20, 20, 3) * 255).astype(np.uint8)
        hist = det._compute_hsv_hist(frame)
        assert hist is not None
        # Normalized to [0, 1]
        assert float(hist.max()) <= 1.0 + 1e-6
        assert float(hist.min()) >= 0.0
