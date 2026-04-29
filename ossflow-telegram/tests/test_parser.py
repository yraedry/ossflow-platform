"""Exhaustive tests for :mod:`telegram_fetcher.parser`."""
from __future__ import annotations

import pytest

from telegram_fetcher.parser import ParsedCaption, parse_caption


# ---------------------------------------------------------------------------
# Canonical cases
# ---------------------------------------------------------------------------


def test_author_dash_title_num_basic():
    r = parse_caption("John Danaher - Leglocks Enter The System 1")
    assert r.author == "John Danaher"
    assert r.title == "Leglocks Enter The System"
    assert r.chapter_num == 1
    assert r.matched_pattern is not None
    assert r.confidence >= 0.9


def test_author_dash_title_semi_subtitle_volume():
    r = parse_caption("John Danaher - Leglocks; Enter The System - Volume 1")
    assert r.author == "John Danaher"
    assert "Leglocks" in r.title
    assert "Enter The System" in r.title
    assert r.chapter_num == 1
    assert r.confidence >= 0.9


def test_author_dash_title_colon_subtitle_volume():
    r = parse_caption("John Danaher - Leglocks: Enter The System - Volume 3")
    assert r.author == "John Danaher"
    assert r.chapter_num == 3
    assert r.confidence >= 0.9


def test_title_by_author_num():
    r = parse_caption("Tripod Passing by Jozef Chen 1")
    assert r.author == "Jozef Chen"
    assert r.title == "Tripod Passing"
    assert r.chapter_num == 1
    assert r.confidence >= 0.9


def test_title_by_author_dash_part():
    r = parse_caption("Half Guard Anthology by Bernardo Faria - Part 4")
    assert r.author == "Bernardo Faria"
    assert r.title == "Half Guard Anthology"
    assert r.chapter_num == 4
    assert r.confidence >= 0.9


def test_author_underscore_title_underscore_num():
    r = parse_caption("Craig_Jones_Z_Guard_Encyclopedia_2")
    assert r.author == "Craig"
    # Title is lowercased -> title-cased
    assert "Jones" in r.title  # because author-underscore captures only first token
    # Actually with pattern (?P<author>[^_]+)_(?P<title>.+)_(?P<chapter>\d+)
    # author = "Craig", title = "Jones_Z_Guard_Encyclopedia"
    assert r.chapter_num == 2


def test_author_underscore_compact():
    r = parse_caption("Danaher_Leglocks_5")
    assert r.author == "Danaher"
    assert r.title == "Leglocks"
    assert r.chapter_num == 5


def test_volume_keyword_variants():
    for kw in ["Volume", "Vol", "Vol.", "Part", "Chapter", "Ch", "Episode", "Ep"]:
        cap = f"John Danaher - Back Attacks - {kw} 7"
        r = parse_caption(cap)
        assert r.chapter_num == 7, f"kw={kw} => {r}"
        assert r.author == "John Danaher"


# ---------------------------------------------------------------------------
# Sanitization: extensions, brackets, parens, whitespace, unicode dashes
# ---------------------------------------------------------------------------


def test_strips_mp4_extension():
    r = parse_caption("John Danaher - Arm Drags 03.mp4")
    assert r.author == "John Danaher"
    assert r.title == "Arm Drags"
    assert r.chapter_num == 3


def test_strips_mkv_extension():
    r = parse_caption("Gordon Ryan - Systematically Attacking The Guard 2.mkv")
    assert r.chapter_num == 2
    assert r.author == "Gordon Ryan"


def test_strips_bracket_info_token():
    r = parse_caption("[BJJFanatics] John Danaher - Arm Drags 03 (HD).mp4")
    assert r.author == "John Danaher"
    assert r.title == "Arm Drags"
    assert r.chapter_num == 3
    # Because we removed two info tokens, confidence should be < 1.0
    assert r.confidence < 1.0
    assert r.confidence >= 0.7


def test_strips_parenthetical_quality():
    r = parse_caption("John Danaher - Leglocks 1 (1080p)")
    assert r.chapter_num == 1
    assert r.title == "Leglocks"


def test_unicode_endash():
    r = parse_caption("John Danaher \u2013 Leglocks 4")
    assert r.author == "John Danaher"
    assert r.chapter_num == 4


def test_unicode_emdash():
    r = parse_caption("gordon ryan\u2014back attacks\u2014volume 5")
    assert r.author == "Gordon Ryan"
    assert "Back Attacks" in r.title
    assert r.chapter_num == 5


def test_trailing_and_leading_whitespace():
    r = parse_caption("   gordon ryan\u2014back attacks\u2014volume 5  ")
    assert r.author == "Gordon Ryan"
    assert r.chapter_num == 5


def test_collapses_internal_whitespace():
    r = parse_caption("John   Danaher   -   Arm  Drags   2")
    assert r.author == "John Danaher"
    assert r.title == "Arm Drags"
    assert r.chapter_num == 2


# ---------------------------------------------------------------------------
# Title Case + acronym preservation
# ---------------------------------------------------------------------------


def test_preserves_bjj_acronym():
    r = parse_caption("John Danaher - BJJ Fundamentals 1")
    assert r.title.startswith("BJJ ")
    assert r.chapter_num == 1


def test_preserves_adcc_acronym():
    r = parse_caption("gordon ryan - ADCC Preparation 3")
    assert "ADCC" in r.title


def test_preserves_dlr_acronym():
    r = parse_caption("craig jones - DLR Guard Attacks 6")
    assert "DLR" in r.title
    assert r.chapter_num == 6


def test_title_cases_lowercase_input():
    r = parse_caption("john danaher - leglocks enter the system 2")
    assert r.author == "John Danaher"
    assert r.title == "Leglocks Enter The System"


# ---------------------------------------------------------------------------
# Imposible / degenerate inputs
# ---------------------------------------------------------------------------


def test_pure_filename_no_metadata():
    r = parse_caption("video.mp4")
    # "video" has no chapter, no dash, shouldn't trigger anything strong.
    assert r.author is None
    assert r.chapter_num is None
    assert r.confidence == 0.0
    assert r.matched_pattern is None


def test_empty_string():
    r = parse_caption("")
    assert r == ParsedCaption(
        author=None,
        title=None,
        chapter_num=None,
        confidence=0.0,
        raw="",
        matched_pattern=None,
    )


def test_only_whitespace():
    r = parse_caption("    \t  \n ")
    assert r.confidence == 0.0
    assert r.author is None
    assert r.title is None


def test_gibberish():
    r = parse_caption("asdf")
    assert r.confidence == 0.0
    assert r.author is None
    assert r.chapter_num is None


def test_trailing_number_only_fallback():
    # Only a bare chapter-ish trailer, not enough to populate author.
    r = parse_caption("unknown clip 7")
    # Pattern 8 (title_trailing_num) will match: title=unknown clip, chapter=7.
    assert r.chapter_num == 7


def test_none_text_safe():
    r = parse_caption(None)  # type: ignore[arg-type]
    assert r.confidence == 0.0


# ---------------------------------------------------------------------------
# Raw passthrough + matched_pattern reporting
# ---------------------------------------------------------------------------


def test_raw_is_preserved():
    text = "[HD] John Danaher - Leglocks 1.mp4"
    r = parse_caption(text)
    assert r.raw == text


def test_matched_pattern_name_reported():
    r = parse_caption("John Danaher - Leglocks 1")
    assert r.matched_pattern == "author_dash_title_num"


def test_matched_pattern_by():
    r = parse_caption("Tripod Passing by Jozef Chen 2")
    assert r.matched_pattern == "title_by_author_num"


# ---------------------------------------------------------------------------
# Confidence contract: >=0.7 for >=80% of the "clean" canonical corpus.
# ---------------------------------------------------------------------------


CLEAN_CORPUS = [
    "John Danaher - Leglocks Enter The System 1",
    "John Danaher - Leglocks; Enter The System - Volume 2",
    "Tripod Passing by Jozef Chen 1",
    "Half Guard Anthology by Bernardo Faria - Part 4",
    "Craig Jones - Z Guard Encyclopedia 3",
    "Gordon Ryan - Systematically Attacking The Guard 7",
    "John Danaher - Back Attacks - Volume 5",
    "John Danaher - BJJ Fundamentals 1",
    "Lachlan Giles - Half Guard Anthology 2",
    "Mikey Musumeci - Ruotolo Chokes 4",
]


def test_confidence_on_clean_corpus():
    high = 0
    for cap in CLEAN_CORPUS:
        r = parse_caption(cap)
        if r.confidence >= 0.7:
            high += 1
    ratio = high / len(CLEAN_CORPUS)
    assert ratio >= 0.8, f"only {ratio:.0%} of clean corpus reached >=0.7 confidence"


def test_dirty_corpus_still_parses():
    """Dirty captions should still extract author/title/chapter even if confidence < 1.0."""
    dirty = [
        "[BJJFanatics] John Danaher - Arm Drags 03 (HD).mp4",
        "   gordon ryan\u2014back attacks\u2014volume 5  ",
        "[1080p] Craig Jones - Just Stand Up 2.mkv",
        "John Danaher - Leglocks 1 (Vol 1).mp4",
    ]
    for cap in dirty:
        r = parse_caption(cap)
        assert r.chapter_num is not None, f"no chapter for {cap!r}: {r}"
        assert r.author is not None, f"no author for {cap!r}: {r}"
        assert r.title is not None, f"no title for {cap!r}: {r}"


# ---------------------------------------------------------------------------
# Pydantic model sanity
# ---------------------------------------------------------------------------


def test_parsed_caption_is_pydantic_model():
    r = parse_caption("John Danaher - Leglocks 1")
    # Pydantic v2: model_dump
    data = r.model_dump()
    assert set(data.keys()) == {
        "author",
        "title",
        "chapter_num",
        "confidence",
        "raw",
        "matched_pattern",
    }


def test_confidence_is_in_unit_interval():
    caps = CLEAN_CORPUS + [
        "[HD] John Danaher - Leglocks 1.mp4",
        "",
        "asdf",
        "video.mp4",
    ]
    for cap in caps:
        r = parse_caption(cap)
        assert 0.0 <= r.confidence <= 1.0, f"{cap!r} => {r.confidence}"


# ---------------------------------------------------------------------------
# Post-parse normalization: author/title cleanup
# (regressions reported in the "Por autor" view: duplicates & leakage)
# ---------------------------------------------------------------------------


def test_author_without_vol_residue():
    r = parse_caption("Gordon Ryan - Systematically Attacking The Guard 1")
    assert r.author == "Gordon Ryan"
    assert r.title == "Systematically Attacking The Guard"
    assert r.chapter_num == 1


def test_author_trailing_vol_is_stripped():
    r = parse_caption("Gordon Ryan Vol - Systematically Attacking The Guard 2")
    assert r.author == "Gordon Ryan"
    assert r.title == "Systematically Attacking The Guard"
    assert r.chapter_num == 2


def test_author_trailing_volume_word_is_stripped():
    r = parse_caption("Gordon Ryan Volume - Back Attacks 3")
    assert r.author == "Gordon Ryan"
    assert r.chapter_num == 3


def test_author_trailing_part_number_is_stripped():
    r = parse_caption("Gordon Ryan Part 2 - Back Attacks 3")
    assert r.author == "Gordon Ryan"
    assert r.chapter_num == 3


def test_author_trailing_dash_vol_n_is_stripped():
    r = parse_caption("Gordon Ryan - Vol 2 - Back Attacks 3")
    # The " - Vol 2" chunk is not an author suffix but a volume segment in the
    # middle; at minimum the final author must be clean (no "Vol" token).
    assert r.author is None or "Vol" not in r.author.split()


def test_standalone_title_with_trailing_number_no_author():
    # "NO Gi Passing 1" alone: author should be None, title the text, chap=1.
    r = parse_caption("NO Gi Passing 1")
    assert r.author is None
    assert r.title is not None
    assert "Passing" in r.title
    assert r.chapter_num == 1


def test_standalone_vol_n_yields_only_chapter():
    r = parse_caption("Vol 3")
    assert r.author is None
    assert r.title is None
    assert r.chapter_num == 3


def test_standalone_volume_n_yields_only_chapter():
    r = parse_caption("Volume 5")
    assert r.author is None
    assert r.title is None
    assert r.chapter_num == 5


def test_standalone_part_n_yields_only_chapter():
    r = parse_caption("Part 7")
    assert r.author is None
    assert r.title is None
    assert r.chapter_num == 7


def test_author_purely_numeric_rejected():
    r = parse_caption("123 - Something 4")
    assert r.author is None


def test_author_too_short_rejected():
    r = parse_caption("AB - Something 4")
    assert r.author is None


def test_title_leading_vol_n_stripped():
    r = parse_caption("John Danaher - Vol 2 Leglocks 3")
    # The "Vol 2" leading piece should not remain inside the title.
    assert r.title is not None
    assert not r.title.lower().startswith("vol ")
    assert "Leglocks" in r.title


def test_author_with_accents_preserved():
    r = parse_caption("João Miyão - Berimbolo 2")
    # Accents preserved in output author (normalization happens only at
    # grouping layer).
    assert r.author is not None
    assert "Miy" in r.author
    assert r.chapter_num == 2


def test_author_vol_dot_suffix_stripped():
    r = parse_caption("Gordon Ryan Vol. - Back Attacks 4")
    assert r.author == "Gordon Ryan"
    assert r.chapter_num == 4


def test_title_case_after_normalization():
    r = parse_caption("gordon ryan vol - back attacks 2")
    assert r.author == "Gordon Ryan"
    assert r.title == "Back Attacks"


def test_acronym_preserved_after_normalization():
    r = parse_caption("Gordon Ryan Vol - ADCC Preparation 1")
    assert r.author == "Gordon Ryan"
    assert "ADCC" in r.title


def test_demote_title_when_author_has_trailing_digit():
    # Heuristic: if an "author" candidate is multi-word with a trailing digit
    # and no title was parsed, treat it as title + chapter.
    # Construct a caption where the only reasonable match is pattern 8 with
    # the shape "NO Gi Passing 1".
    r = parse_caption("NO Gi Passing 1")
    assert r.author is None
    assert r.chapter_num == 1
    assert r.title is not None


@pytest.mark.parametrize(
    "caption, expected_chap",
    [
        ("John Danaher - Leglocks 1", 1),
        ("John Danaher - Leglocks 2", 2),
        ("John Danaher - Leglocks 10", 10),
        ("John Danaher - Leglocks 99", 99),
        ("John Danaher - Leglocks 123", 123),
    ],
)
def test_chapter_number_parsing(caption, expected_chap):
    r = parse_caption(caption)
    assert r.chapter_num == expected_chap
