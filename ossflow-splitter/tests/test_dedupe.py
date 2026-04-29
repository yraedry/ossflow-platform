"""Tests for consecutive-title deduplication in ChapterDetector."""

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


class TestAreTitlesDuplicate:
    def test_identical_titles_are_duplicate(self):
        det = _make_detector(Config())
        assert det._are_titles_duplicate("DESIGN", "DESIGN") is True

    def test_substring_is_duplicate(self):
        det = _make_detector(Config())
        assert det._are_titles_duplicate("DESIGN", "DESIGN PRINCIPLES") is True
        assert det._are_titles_duplicate("UPPER BODY", "UPPER") is True

    def test_near_match_within_threshold(self):
        det = _make_detector(Config(dedupe_similarity_threshold=0.25))
        # Small OCR drift -- should be treated as dupe.
        assert det._are_titles_duplicate("DESIGN", "DES1GN") is True

    def test_clearly_different_titles_are_not_duplicate(self):
        det = _make_detector(Config())
        assert det._are_titles_duplicate("ARM DRAG", "LEG LOCK") is False

    def test_empty_is_not_duplicate(self):
        det = _make_detector(Config())
        assert det._are_titles_duplicate("", "FOO") is False
        assert det._are_titles_duplicate("FOO", "") is False

    def test_case_and_punctuation_insensitive(self):
        det = _make_detector(Config())
        assert det._are_titles_duplicate("Design,", "design") is True


class TestDedupeConsecutive:
    def test_merges_two_consecutive_design(self):
        det = _make_detector(Config())
        chapters = [
            Chapter(0.0, 100.0, "INTRO"),
            Chapter(100.0, 230.0, "DESIGN"),
            Chapter(230.0, 370.0, "DESIGN"),  # duplicate, 140s after
            Chapter(370.0, 500.0, "FINISH"),
        ]
        out = det._dedupe_consecutive(chapters)
        assert len(out) == 3
        assert out[0].title == "INTRO"
        assert out[1].title == "DESIGN"
        assert out[1].start == 100.0
        assert out[1].end == 370.0
        assert out[2].title == "FINISH"

    def test_no_dupes_passes_through(self):
        det = _make_detector(Config())
        chapters = [
            Chapter(0.0, 60.0, "A"),
            Chapter(60.0, 120.0, "B"),
            Chapter(120.0, 200.0, "C"),
        ]
        out = det._dedupe_consecutive(chapters)
        assert len(out) == 3
        assert [c.title for c in out] == ["A", "B", "C"]

    def test_single_chapter_unchanged(self):
        det = _make_detector(Config())
        chapters = [Chapter(0.0, 100.0, "ONLY")]
        out = det._dedupe_consecutive(chapters)
        assert out == chapters

    def test_empty_list(self):
        det = _make_detector(Config())
        assert det._dedupe_consecutive([]) == []

    def test_chain_of_three_duplicates_merges_to_one(self):
        det = _make_detector(Config())
        chapters = [
            Chapter(0.0, 50.0, "X"),
            Chapter(50.0, 120.0, "X"),
            Chapter(120.0, 180.0, "X"),
        ]
        out = det._dedupe_consecutive(chapters)
        assert len(out) == 1
        assert out[0].start == 0.0
        assert out[0].end == 180.0

    def test_default_threshold(self):
        assert Config().dedupe_similarity_threshold == 0.25
