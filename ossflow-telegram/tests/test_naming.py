"""Tests for naming.py — Windows sanitization + oracle-compatible chapter filename."""
from __future__ import annotations

import re

import pytest

from telegram_fetcher.naming import (
    MAX_SEGMENT_LEN,
    chapter_filename,
    instructional_dirname,
    sanitize_for_windows,
)


class TestSanitize:
    def test_removes_forbidden_chars(self) -> None:
        assert sanitize_for_windows('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"

    def test_strips_trailing_dots_and_spaces(self) -> None:
        assert sanitize_for_windows("hello.  ") == "hello"

    def test_collapses_whitespace(self) -> None:
        assert sanitize_for_windows("a    b\t c") == "a b c"

    def test_reserved_names_prefixed(self) -> None:
        assert sanitize_for_windows("CON").startswith("_")
        assert sanitize_for_windows("com1").startswith("_") or sanitize_for_windows("com1") == "_com1"

    def test_empty_input_is_replaced(self) -> None:
        assert sanitize_for_windows("") == "_"
        assert sanitize_for_windows("   ") == "_"

    def test_truncates_long_segment(self) -> None:
        out = sanitize_for_windows("a" * (MAX_SEGMENT_LEN + 50))
        assert len(out) <= MAX_SEGMENT_LEN

    def test_none_input(self) -> None:
        assert sanitize_for_windows(None) == ""  # type: ignore[arg-type]


class TestDirname:
    def test_basic(self) -> None:
        assert instructional_dirname("John Danaher", "Leglocks") == "John Danaher - Leglocks"

    def test_colon_replaced(self) -> None:
        got = instructional_dirname("John Danaher", "Leglocks: Enter the System")
        assert ":" not in got
        assert "Leglocks_ Enter the System" in got


class TestChapterFilename:
    def test_no_space_before_number(self) -> None:
        name = chapter_filename("John Danaher", "Leglocks", 3, ".mp4")
        assert name == "John Danaher - Leglocks3.mp4"
        stem = name.rsplit(".", 1)[0]
        # The oracle splitter contract: trailing digits must match chapter_num.
        m = re.search(r"(\d+)\s*$", stem)
        assert m is not None
        assert m.group(1) == "3"

    def test_ext_without_dot_is_normalized(self) -> None:
        name = chapter_filename("A", "B", 1, "mkv")
        assert name.endswith(".mkv")

    def test_regex_oracle_compat_multi_digit(self) -> None:
        name = chapter_filename("Craig Jones", "Passing", 12, ".mp4")
        stem = name.rsplit(".", 1)[0]
        assert re.search(r"(\d+)\s*$", stem).group(1) == "12"

    def test_truncation_preserves_chapter_and_ext(self) -> None:
        long_title = "x" * 500
        name = chapter_filename("A", long_title, 7, ".mp4")
        assert name.endswith("7.mp4")
        assert len(name) <= MAX_SEGMENT_LEN

    def test_forbidden_chars_stripped_from_author(self) -> None:
        name = chapter_filename("A/B", "T", 1, ".mp4")
        assert "/" not in name
