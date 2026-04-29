"""Tests for subtitle_generator.writer.SubtitleWriter."""

import pytest

from subtitle_generator.config import SubtitleConfig
from subtitle_generator.timestamp_fixer import TimestampFixer
from subtitle_generator.writer import SubtitleWriter


@pytest.fixture
def writer():
    """SubtitleWriter with default config."""
    cfg = SubtitleConfig()
    fixer = TimestampFixer(cfg)
    return SubtitleWriter(cfg, fixer)


@pytest.fixture
def writer_narrow():
    """SubtitleWriter with narrow line width (20 chars)."""
    cfg = SubtitleConfig(max_chars_per_line=20)
    fixer = TimestampFixer(cfg)
    return SubtitleWriter(cfg, fixer)


# ---------------------------------------------------------------------------
# _format_lines
# ---------------------------------------------------------------------------

class TestFormatLines:
    """Test punctuation-aware line breaking."""

    def test_short_text_no_break(self, writer):
        text = "Hello world."
        result = writer._format_lines(text)
        assert "\n" not in result
        assert result == "Hello world."

    def test_two_lines_for_long_text(self, writer):
        # 42-char limit; this text exceeds it
        text = "The arm drag is a fundamental technique in Brazilian Jiu-Jitsu grappling."
        result = writer._format_lines(text)
        lines = result.split("\n")
        assert len(lines) == 2
        # Each line should be <= max_chars_per_line
        for line in lines:
            assert len(line) <= 42

    def test_single_word_no_break(self, writer):
        text = "Supercalifragilisticexpialidocious"
        result = writer._format_lines(text)
        assert "\n" not in result


class TestFindBestBreak:
    """Test _find_best_break punctuation preferences."""

    def test_prefers_sentence_end(self, writer):
        # Break after sentence-ending punctuation
        words = ["First sentence.", "Second", "sentence", "here."]
        idx = writer._find_best_break(words, max_chars=42)
        # Should prefer breaking after "sentence." (index 1)
        assert idx == 1

    def test_prefers_clause_over_conjunction(self, writer):
        words = ["First", "clause,", "and", "then", "more."]
        idx = writer._find_best_break(words, max_chars=42)
        # Clause end (after comma) has score 10, conjunction has 20
        # But balance also matters; "First clause," vs "and then more."
        # idx=2 means break after "clause,"
        assert idx == 2

    def test_prefers_conjunction_over_nothing(self, writer):
        words = ["word1", "word2", "and", "word3", "word4"]
        idx = writer._find_best_break(words, max_chars=42)
        # "and" is a conjunction -> break before it preferred over arbitrary break
        # idx=2 means line1="word1 word2", line2="and word3 word4"
        assert idx in (2, 3)  # either before or after conjunction area

    def test_balanced_lines_preferred(self, writer):
        words = ["aa", "bb", "cc", "dd"]
        idx = writer._find_best_break(words, max_chars=42)
        # All break points have same punct_bonus=30, so balance wins -> idx=2
        assert idx == 2


# ---------------------------------------------------------------------------
# _build_blocks
# ---------------------------------------------------------------------------

class TestBuildBlocks:
    """Test that words are grouped into blocks respecting character limits."""

    def test_build_blocks_respects_max_chars(self, writer):
        # Each word is short; total chars per block should not exceed max_chars * max_lines
        words = [
            {"word": f"word{i}", "start": float(i), "end": float(i) + 0.5}
            for i in range(50)
        ]
        blocks = writer._build_blocks(words)
        max_total = writer.config.max_chars_per_line * writer.config.max_lines
        for block in blocks:
            # The raw text (before formatting) should be close to the limit
            raw_text = block["text"].replace("\n", " ")
            # Allow small overshoot due to word boundaries
            assert len(raw_text) <= max_total + 20

    def test_build_blocks_start_end_times(self, writer):
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
        blocks = writer._build_blocks(words)
        assert len(blocks) == 1
        assert blocks[0]["start"] == 0.0
        assert blocks[0]["end"] == 1.0

    def test_build_blocks_empty_words_skipped(self, writer):
        words = [
            {"word": "", "start": 0.0, "end": 0.5},
            {"word": "hello", "start": 0.6, "end": 1.0},
            {"word": "  ", "start": 1.1, "end": 1.5},
        ]
        blocks = writer._build_blocks(words)
        assert len(blocks) == 1
        assert "hello" in blocks[0]["text"]

    def test_build_blocks_many_words_creates_multiple_blocks(self, writer_narrow):
        words = [
            {"word": f"longword{i}", "start": float(i), "end": float(i) + 0.5}
            for i in range(20)
        ]
        blocks = writer_narrow._build_blocks(words)
        assert len(blocks) > 1
