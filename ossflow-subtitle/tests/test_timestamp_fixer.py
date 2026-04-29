"""Tests for subtitle_generator.timestamp_fixer.TimestampFixer."""

import pytest

from subtitle_generator.config import SubtitleConfig
from subtitle_generator.timestamp_fixer import TimestampFixer


@pytest.fixture
def fixer():
    """TimestampFixer with default config."""
    return TimestampFixer(SubtitleConfig())


# ---------------------------------------------------------------------------
# _interpolate_null_timestamps
# ---------------------------------------------------------------------------

class TestInterpolateNullTimestamps:
    """Test interpolation of null start/end word timestamps."""

    def test_interpolate_null_start(self, fixer):
        words = [
            {"word": "hello", "start": None, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
        result = fixer._interpolate_null_timestamps(words)
        assert result[0]["start"] is not None
        assert isinstance(result[0]["start"], float)

    def test_interpolate_null_end(self, fixer):
        words = [
            {"word": "hello", "start": 0.0, "end": None},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
        result = fixer._interpolate_null_timestamps(words)
        assert result[0]["end"] is not None
        assert isinstance(result[0]["end"], float)

    def test_all_nulls_get_values(self, fixer):
        words = [
            {"word": "a", "start": 0.0, "end": 0.5},
            {"word": "b", "start": None, "end": None},
            {"word": "c", "start": 2.0, "end": 2.5},
        ]
        result = fixer._interpolate_null_timestamps(words)
        assert result[1]["start"] is not None
        assert result[1]["end"] is not None
        # Interpolated values should be between neighbors
        assert result[1]["start"] >= 0.0
        assert result[1]["end"] <= 2.5

    def test_end_never_before_start(self, fixer):
        words = [
            {"word": "only", "start": None, "end": None},
        ]
        result = fixer._interpolate_null_timestamps(words)
        assert result[0]["end"] >= result[0]["start"]


# ---------------------------------------------------------------------------
# _remove_word_overlaps
# ---------------------------------------------------------------------------

class TestRemoveWordOverlaps:
    """Test that word-level overlaps are resolved by splitting the difference."""

    def test_remove_word_overlaps(self, fixer):
        words = [
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "world", "start": 0.5, "end": 1.5},  # overlaps by 0.5s
        ]
        result = fixer._remove_word_overlaps(words)
        # After fix: words[0].end == words[1].start == midpoint of (1.0, 0.5) = 0.75
        assert result[0]["end"] == result[1]["start"]
        assert result[0]["end"] == pytest.approx(0.75)

    def test_no_overlap_unchanged(self, fixer):
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
        result = fixer._remove_word_overlaps(words)
        assert result[0]["end"] == 0.5
        assert result[1]["start"] == 0.6


# ---------------------------------------------------------------------------
# _remove_overlaps (subtitle-level)
# ---------------------------------------------------------------------------

class TestRemoveSubtitleOverlaps:
    """Test that subtitle-level overlaps are resolved."""

    def test_remove_subtitle_overlaps(self, fixer):
        subs = [
            {"start": 0.0, "end": 3.0, "text": "First"},
            {"start": 2.0, "end": 5.0, "text": "Second"},  # overlaps by 1s
        ]
        result = fixer._remove_overlaps(subs)
        assert result[0]["end"] == result[1]["start"]
        assert result[0]["end"] == pytest.approx(2.5)

    def test_no_overlap_unchanged(self, fixer):
        subs = [
            {"start": 0.0, "end": 2.0, "text": "First"},
            {"start": 3.0, "end": 5.0, "text": "Second"},
        ]
        result = fixer._remove_overlaps(subs)
        assert result[0]["end"] == 2.0
        assert result[1]["start"] == 3.0


# ---------------------------------------------------------------------------
# _fill_small_gaps
# ---------------------------------------------------------------------------

class TestFillSmallGaps:
    """Test that small gaps (<100ms) are filled by extending previous subtitle."""

    def test_fill_small_gaps(self, fixer):
        subs = [
            {"start": 0.0, "end": 2.0, "text": "First"},
            {"start": 2.05, "end": 4.0, "text": "Second"},  # 50ms gap
        ]
        result = fixer._fill_small_gaps(subs)
        # Gap of 50ms < 100ms threshold -> filled
        assert result[0]["end"] == result[1]["start"]

    def test_large_gap_not_filled(self, fixer):
        subs = [
            {"start": 0.0, "end": 2.0, "text": "First"},
            {"start": 3.0, "end": 5.0, "text": "Second"},  # 1s gap
        ]
        result = fixer._fill_small_gaps(subs)
        assert result[0]["end"] == 2.0

    def test_zero_gap_not_affected(self, fixer):
        subs = [
            {"start": 0.0, "end": 2.0, "text": "First"},
            {"start": 2.0, "end": 4.0, "text": "Second"},  # no gap
        ]
        result = fixer._fill_small_gaps(subs)
        assert result[0]["end"] == 2.0


# ---------------------------------------------------------------------------
# _clamp_durations
# ---------------------------------------------------------------------------

class TestClampDurations:
    """Test clamping of subtitle durations to [min, max]."""

    def test_clamp_too_short(self, fixer):
        subs = [{"start": 0.0, "end": 0.2, "text": "Short"}]  # 0.2s < 0.5s min
        result = fixer._clamp_durations(subs)
        duration = result[0]["end"] - result[0]["start"]
        assert duration == pytest.approx(0.5)

    def test_clamp_too_long(self, fixer):
        subs = [{"start": 0.0, "end": 20.0, "text": "Long"}]  # 20s > 7s max
        result = fixer._clamp_durations(subs)
        duration = result[0]["end"] - result[0]["start"]
        assert duration == pytest.approx(7.0)

    def test_normal_duration_unchanged(self, fixer):
        subs = [{"start": 0.0, "end": 3.0, "text": "Normal"}]  # 3s within [0.7, 7.0]
        result = fixer._clamp_durations(subs)
        assert result[0]["end"] == 3.0

    def test_clamp_preserves_start(self, fixer):
        subs = [{"start": 10.0, "end": 10.1, "text": "Short"}]
        result = fixer._clamp_durations(subs)
        assert result[0]["start"] == 10.0
        assert result[0]["end"] == pytest.approx(10.5)


# ---------------------------------------------------------------------------
# End >= Start enforcement
# ---------------------------------------------------------------------------

class TestEndGreaterThanStart:
    """After fix_words, end should always be >= start for every word."""

    def test_end_greater_than_start_enforced(self, fixer):
        words = [
            {"word": "a", "start": 1.0, "end": 0.5},  # end < start
            {"word": "b", "start": 2.0, "end": 2.5},
        ]
        result = fixer._interpolate_null_timestamps(words)
        for w in result:
            assert w["end"] >= w["start"], f"end ({w['end']}) < start ({w['start']})"


# ---------------------------------------------------------------------------
# fix_words / fix_subtitles integration
# ---------------------------------------------------------------------------

class TestFixWordsIntegration:
    """Integration tests for fix_words."""

    def test_fix_words_handles_empty(self, fixer):
        assert fixer.fix_words([]) == []

    def test_fix_subtitles_handles_empty(self, fixer):
        assert fixer.fix_subtitles([]) == []

    def test_fix_words_full_pipeline(self, fixer):
        words = [
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "world", "start": None, "end": None},
            {"word": "test", "start": 2.0, "end": 3.0},
        ]
        result = fixer.fix_words(words)
        # All words should have valid timestamps
        for w in result:
            assert w["start"] is not None
            assert w["end"] is not None
            assert w["end"] >= w["start"]
