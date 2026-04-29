"""Tests for OCR blacklist + title-confidence gating helpers in ChapterDetector."""

import unittest.mock as mock

import pytest

from chapter_splitter.config import Config

# cv2 is imported at module level; mock before importing detector.
cv2_mock = mock.MagicMock()
with mock.patch.dict("sys.modules", {"cv2": cv2_mock}):
    from chapter_splitter.detection.detector import ChapterDetector


def _make_detector(cfg: Config) -> ChapterDetector:
    """Build a detector instance without touching OCR/Audio/memory."""
    det = object.__new__(ChapterDetector)
    det.cfg = cfg
    det.ocr = None
    det.audio = None
    det.memory = None
    det.stability = None
    return det


class TestNormalizeForMatch:
    def test_lowercases(self):
        assert ChapterDetector._normalize_for_match("BJJ Fanatics") == "bjj fanatics"

    def test_strips_punctuation(self):
        assert ChapterDetector._normalize_for_match("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert ChapterDetector._normalize_for_match("  foo   bar  ") == "foo bar"

    def test_empty_returns_empty(self):
        assert ChapterDetector._normalize_for_match("") == ""
        assert ChapterDetector._normalize_for_match(None) == ""


class TestIsBlacklisted:
    def test_bjjfanatics_watermark_detected(self):
        det = _make_detector(Config())
        assert det._is_blacklisted("GUNDO HFANATICS 0") is True
        assert det._is_blacklisted("IFANATICS 0") is True
        assert det._is_blacklisted("DESIGN TMTICS") is False  # no blacklist token

    def test_case_insensitive(self):
        det = _make_detector(Config())
        assert det._is_blacklisted("bjjfanatics") is True
        assert det._is_blacklisted("BJJFANATICS") is True
        assert det._is_blacklisted("Bjj Fanatics") is True

    def test_clean_title_not_blacklisted(self):
        det = _make_detector(Config())
        assert det._is_blacklisted("UPPER BODY CONTROL") is False
        assert det._is_blacklisted("ARM DRAG SYSTEM") is False

    def test_custom_blacklist_via_config(self):
        det = _make_detector(Config(ocr_blacklist=("sponsor",)))
        assert det._is_blacklisted("sponsor message") is True
        assert det._is_blacklisted("bjjfanatics") is False

    def test_empty_text_not_blacklisted(self):
        det = _make_detector(Config())
        assert det._is_blacklisted("") is False


class TestMinTitleChars:
    """Simulate the length gate applied to candidate titles in the detector."""

    def test_gate_rejects_short_titles(self):
        cfg = Config(ocr_min_title_chars=4)
        short = "0"
        assert len(short.strip()) < cfg.ocr_min_title_chars

    def test_gate_accepts_long_enough_titles(self):
        cfg = Config(ocr_min_title_chars=4)
        assert len("UPPER".strip()) >= cfg.ocr_min_title_chars


class TestTitleConfidenceGate:
    def test_default_title_confidence_is_higher_than_generic(self):
        cfg = Config()
        assert cfg.ocr_title_confidence_min > cfg.ocr_confidence_min

    def test_generic_ocr_still_055(self):
        cfg = Config()
        assert cfg.ocr_confidence_min == 0.55
        assert cfg.ocr_title_confidence_min == 0.75
