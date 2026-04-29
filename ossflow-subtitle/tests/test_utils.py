"""Tests for subtitle_generator.utils.format_timestamp."""

import pytest

from subtitle_generator.utils import format_timestamp


class TestFormatTimestamp:
    """Test SRT timestamp formatting."""

    def test_zero(self):
        assert format_timestamp(0.0) == "00:00:00,000"

    def test_one_second(self):
        assert format_timestamp(1.0) == "00:00:01,000"

    def test_fractional_seconds(self):
        assert format_timestamp(1.5) == "00:00:01,500"

    def test_minutes(self):
        assert format_timestamp(65.0) == "00:01:05,000"

    def test_hours(self):
        assert format_timestamp(3661.0) == "01:01:01,000"

    def test_milliseconds_precision(self):
        assert format_timestamp(0.123) == "00:00:00,123"

    def test_large_value(self):
        # 10 hours + 30 min + 45 sec + 678ms
        secs = 10 * 3600 + 30 * 60 + 45 + 0.678
        assert format_timestamp(secs) == "10:30:45,678"

    def test_negative_clamped_to_zero(self):
        assert format_timestamp(-5.0) == "00:00:00,000"

    def test_none_clamped_to_zero(self):
        assert format_timestamp(None) == "00:00:00,000"

    def test_small_milliseconds(self):
        assert format_timestamp(0.001) == "00:00:00,001"

    def test_just_under_one_second(self):
        result = format_timestamp(0.999)
        assert result == "00:00:00,999"
