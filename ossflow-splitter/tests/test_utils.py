"""Tests for chapter_splitter.utils: sanitize_filename and extract_season_number."""

import pytest

from chapter_splitter.utils import sanitize_filename, extract_season_number


class TestSanitizeFilename:
    """Test sanitize_filename with various inputs."""

    def test_removes_backslash(self):
        assert "\\" not in sanitize_filename("foo\\bar")

    def test_removes_slash(self):
        assert "/" not in sanitize_filename("foo/bar")

    def test_removes_asterisk(self):
        assert "*" not in sanitize_filename("foo*bar")

    def test_removes_question_mark(self):
        assert "?" not in sanitize_filename("foo?bar")

    def test_removes_colon(self):
        assert ":" not in sanitize_filename("foo:bar")

    def test_removes_quotes(self):
        assert '"' not in sanitize_filename('foo"bar')

    def test_removes_angle_brackets(self):
        result = sanitize_filename("foo<bar>baz")
        assert "<" not in result
        assert ">" not in result

    def test_removes_pipe(self):
        assert "|" not in sanitize_filename("foo|bar")

    def test_removes_caret_and_tilde(self):
        result = sanitize_filename("foo^bar~baz")
        assert "^" not in result
        assert "~" not in result

    def test_removes_copyright_and_registered(self):
        result = sanitize_filename("foo\xa9bar\xae")
        assert "\xa9" not in result
        assert "\xae" not in result

    def test_removes_semicolon_and_braces(self):
        result = sanitize_filename("foo;bar{baz}")
        assert ";" not in result
        assert "{" not in result
        assert "}" not in result

    def test_replaces_newlines_with_space(self):
        assert sanitize_filename("foo\nbar") == "foo bar"
        assert sanitize_filename("foo\rbar") == "foo bar"

    def test_collapses_multiple_spaces(self):
        assert sanitize_filename("foo   bar") == "foo bar"

    def test_strips_leading_trailing_whitespace(self):
        assert sanitize_filename("  hello  ") == "hello"

    def test_preserves_normal_text(self):
        assert sanitize_filename("Chapter 1 - Introduction") == "Chapter 1 - Introduction"

    def test_unicode_letters_preserved(self):
        # Letters like accented chars should survive (not in the removal regex)
        assert sanitize_filename("cafe") == "cafe"

    def test_empty_string(self):
        assert sanitize_filename("") == ""

    def test_long_string_not_truncated(self):
        # sanitize_filename does NOT truncate; it only cleans characters
        long = "A" * 300
        assert sanitize_filename(long) == long


class TestExtractSeasonNumber:
    """Test extract_season_number with various filename patterns."""

    def test_vol_pattern(self):
        assert extract_season_number("Vol 3 - Techniques.mp4", fallback=1) == 3

    def test_volume_pattern(self):
        assert extract_season_number("Volume 12 content.mkv", fallback=1) == 12

    def test_part_pattern(self):
        assert extract_season_number("Part 2 - Guard.mp4", fallback=1) == 2

    def test_disc_pattern(self):
        assert extract_season_number("Disc 5 extras.avi", fallback=1) == 5

    def test_disk_pattern(self):
        assert extract_season_number("Disk 7.mkv", fallback=1) == 7

    def test_case_insensitive(self):
        assert extract_season_number("VOLUME 4.mp4", fallback=1) == 4

    def test_separator_dash(self):
        assert extract_season_number("Vol-2.mp4", fallback=1) == 2

    def test_separator_dot(self):
        assert extract_season_number("Vol.3.mp4", fallback=1) == 3

    def test_separator_underscore(self):
        assert extract_season_number("Part_4.mp4", fallback=1) == 4

    def test_no_match_returns_fallback(self):
        assert extract_season_number("random_video.mp4", fallback=99) == 99

    def test_fallback_zero(self):
        assert extract_season_number("no_match.mp4", fallback=0) == 0

    def test_first_match_used(self):
        # regex finds the first match
        result = extract_season_number("Vol 1 Part 2.mp4", fallback=0)
        assert result == 1
