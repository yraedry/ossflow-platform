"""Tests for subtitle_generator.validator.SubtitleValidator."""

import pytest

from subtitle_generator.config import SubtitleConfig
from subtitle_generator.validator import SubtitleValidator


@pytest.fixture
def validator():
    """SubtitleValidator with default config."""
    return SubtitleValidator(SubtitleConfig())


# ---------------------------------------------------------------------------
# Clean subtitles
# ---------------------------------------------------------------------------

class TestValidatesCleanSubtitles:
    """Clean subtitles should produce no issues."""

    def test_validates_clean_subtitles(self, validator):
        subs = [
            {"start": 0.0, "end": 2.0, "text": "Hello world."},
            {"start": 2.5, "end": 4.5, "text": "This is a test."},
            {"start": 5.0, "end": 7.0, "text": "Everything is fine."},
        ]
        report = validator.validate(subs)
        assert report["issues"] == []
        assert report["large_gaps"] == []
        assert report["total_segments"] == 3


# ---------------------------------------------------------------------------
# Empty text detection
# ---------------------------------------------------------------------------

class TestDetectsEmptyText:
    """Subtitles with empty text should be flagged."""

    def test_detects_empty_text(self, validator):
        subs = [
            {"start": 0.0, "end": 2.0, "text": "Valid text."},
            {"start": 3.0, "end": 5.0, "text": ""},
            {"start": 6.0, "end": 8.0, "text": "   "},
        ]
        report = validator.validate(subs)
        empty_issues = [i for i in report["issues"] if "empty text" in i]
        assert len(empty_issues) >= 1


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

class TestDetectsOverlaps:
    """Overlapping subtitles should be flagged."""

    def test_detects_overlaps(self, validator):
        subs = [
            {"start": 0.0, "end": 3.0, "text": "First."},
            {"start": 2.0, "end": 5.0, "text": "Second."},  # overlaps by 1s
        ]
        report = validator.validate(subs)
        overlap_issues = [i for i in report["issues"] if "overlaps" in i]
        assert len(overlap_issues) == 1

    def test_no_overlap_no_issue(self, validator):
        subs = [
            {"start": 0.0, "end": 2.0, "text": "First."},
            {"start": 3.0, "end": 5.0, "text": "Second."},
        ]
        report = validator.validate(subs)
        overlap_issues = [i for i in report["issues"] if "overlaps" in i]
        assert len(overlap_issues) == 0


# ---------------------------------------------------------------------------
# Large gap detection
# ---------------------------------------------------------------------------

class TestDetectsLargeGaps:
    """Gaps larger than gap_warn_threshold should be reported."""

    def test_detects_large_gaps(self, validator):
        subs = [
            {"start": 0.0, "end": 2.0, "text": "First."},
            {"start": 10.0, "end": 12.0, "text": "After long gap."},  # 8s gap > 5s threshold
        ]
        report = validator.validate(subs)
        assert len(report["large_gaps"]) == 1
        assert report["large_gaps"][0]["gap_seconds"] == pytest.approx(8.0)

    def test_small_gap_not_flagged(self, validator):
        subs = [
            {"start": 0.0, "end": 2.0, "text": "First."},
            {"start": 3.0, "end": 5.0, "text": "Second."},  # 1s gap < 5s threshold
        ]
        report = validator.validate(subs)
        assert len(report["large_gaps"]) == 0


# ---------------------------------------------------------------------------
# Coverage calculation
# ---------------------------------------------------------------------------

class TestCoverageCalculation:
    """Test that coverage percentage is correctly computed."""

    def test_coverage_calculation(self, validator):
        subs = [
            {"start": 0.0, "end": 5.0, "text": "First."},
            {"start": 5.0, "end": 10.0, "text": "Second."},
        ]
        report = validator.validate(subs)
        # Total subtitle time = 10s, span = 10s -> 100%
        assert report["coverage_percent"] == pytest.approx(100.0)
        assert report["total_subtitle_time"] == pytest.approx(10.0)
        assert report["span"] == pytest.approx(10.0)

    def test_partial_coverage(self, validator):
        subs = [
            {"start": 0.0, "end": 2.0, "text": "First."},
            {"start": 8.0, "end": 10.0, "text": "Second."},
        ]
        report = validator.validate(subs)
        # Total subtitle time = 4s, span = 10s -> 40%
        assert report["coverage_percent"] == pytest.approx(40.0)

    def test_empty_subtitles(self, validator):
        report = validator.validate([])
        assert report["total_segments"] == 0
        assert report["coverage_percent"] == 0
        assert report["span"] == 0

    def test_single_subtitle_coverage(self, validator):
        subs = [{"start": 5.0, "end": 8.0, "text": "Only one."}]
        report = validator.validate(subs)
        assert report["coverage_percent"] == pytest.approx(100.0)
        assert report["total_subtitle_time"] == pytest.approx(3.0)
