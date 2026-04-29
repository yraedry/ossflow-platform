"""Caption parser for Telegram media messages.

Tries a prioritized list of regex patterns. The first one that matches wins.
Outputs a ParsedCaption with author/title/chapter_num plus a confidence score.

Design notes:
- We first "sanitize" the input: strip file extensions, remove informational
  parenthetical / bracketed tokens like ``[1080p]``, ``(HD)``, ``(Vol 1)``,
  collapse duplicated whitespace, normalize unicode dashes (``–``, ``—``) to
  ASCII ``-``.
- We keep track of tokens that we removed or that did not get mapped to a
  field, and translate that into a <1.0 confidence.
- Title is rendered in Title Case, but all-caps acronyms of length <=4 are
  preserved (BJJ, MMA, DLR, ADCC, etc.).
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ParsedCaption(BaseModel):
    author: Optional[str] = None
    title: Optional[str] = None
    chapter_num: Optional[int] = None
    confidence: float = 0.0  # 0.0 .. 1.0
    raw: str = ""
    matched_pattern: Optional[str] = None


# ---------------------------------------------------------------------------
# Sanitization helpers
# ---------------------------------------------------------------------------


_VIDEO_EXT_RE = re.compile(r"\.(mp4|mkv|mov|avi|webm|m4v|wmv|flv)$", re.IGNORECASE)
_UNICODE_DASHES = {"\u2013": "-", "\u2014": "-", "\u2212": "-"}
_INFO_BRACKET_RE = re.compile(r"\[[^\[\]]*\]")
# Parenthetical blocks that look informational (resolution, quality, vol tags,
# language, etc.). We treat all () groups as informational noise -- titles that
# genuinely contain parens are rare in this corpus.
_INFO_PAREN_RE = re.compile(r"\([^()]*\)")
_MULTI_WS_RE = re.compile(r"\s+")

_ACRONYM_WHITELIST = {
    "BJJ",
    "MMA",
    "DLR",
    "ADCC",
    "UFC",
    "NOGI",
    "IBJJF",  # len 5, still commonly an acronym; kept via explicit whitelist
}


def _normalize_dashes(text: str) -> str:
    for src, dst in _UNICODE_DASHES.items():
        text = text.replace(src, dst)
    return text


def _strip_extension(text: str) -> str:
    return _VIDEO_EXT_RE.sub("", text).strip()


def _strip_info_tokens(text: str) -> tuple[str, list[str]]:
    """Remove [..] and (..) blocks. Return cleaned text and list of removed tokens."""
    removed: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        removed.append(match.group(0))
        return " "

    text = _INFO_BRACKET_RE.sub(_collect, text)
    text = _INFO_PAREN_RE.sub(_collect, text)
    return text, removed


def _collapse_whitespace(text: str) -> str:
    return _MULTI_WS_RE.sub(" ", text).strip()


def _sanitize(raw: str) -> tuple[str, list[str]]:
    """Return (cleaned_text, list_of_removed_info_tokens)."""
    text = raw
    text = _normalize_dashes(text)
    text = _strip_extension(text)
    text, removed = _strip_info_tokens(text)
    text = _collapse_whitespace(text)
    return text, removed


def _title_case(text: str) -> str:
    """Title-case a string while preserving short all-caps acronyms."""
    if not text:
        return text
    words = text.split(" ")
    out: list[str] = []
    for w in words:
        if not w:
            continue
        # Preserve a token if it's an all-caps acronym of length <= 4 or in
        # the explicit whitelist.
        stripped = re.sub(r"[^A-Za-z]", "", w)
        if stripped and stripped.isupper() and (len(stripped) <= 4 or stripped in _ACRONYM_WHITELIST):
            out.append(w)
            continue
        # Title-case, but keep internal apostrophes / hyphens nice.
        out.append(w[:1].upper() + w[1:].lower() if len(w) > 1 else w.upper())
    return " ".join(out)


def _clean_field(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = text.strip().strip("-_:;,. \t")
    t = _collapse_whitespace(t)
    return t or None


# ---------------------------------------------------------------------------
# Post-parse normalization (author / title cleanup)
# ---------------------------------------------------------------------------


# Trailing residues we strip off authors (after a match). The volume keywords
# may leak in when a caption has the shape "Author Vol - Title 2" and the
# regex grabs "Author Vol" as author.
_AUTHOR_TRAILING_RE = re.compile(
    r"""\s*(?:-\s*)?(?:Volume|Vol|Part|Chapter|Ch|Episode|Ep)\.?
        (?:\s*\d+)?\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# Leading residues we strip off titles.
_TITLE_LEADING_RE = re.compile(
    r"""^\s*(?:Volume|Vol|Part|Chapter|Ch|Episode|Ep)\.?\s*\d+\s*[-:;]?\s*""",
    re.IGNORECASE | re.VERBOSE,
)

# A bare-keyword title, i.e. just "Vol" or "Part" with no accompanying text.
_TITLE_BARE_KEYWORD_RE = re.compile(
    r"^\s*(?:Volume|Vol|Part|Chapter|Ch|Episode|Ep)\.?\s*$",
    re.IGNORECASE,
)

# Residual parenthetical / bracket leftovers that survived sanitization
# (e.g. unmatched brackets).
_RESIDUAL_BRACKETS_RE = re.compile(r"[\[\]()]")


def _normalize_author(raw: Optional[str]) -> Optional[str]:
    """Normalize a raw author string extracted by a regex.

    - Strips trailing "Vol", "Volume", "Volume N", "Part N", "- Vol N" etc.
    - Returns None if the result is numeric only or <= 2 chars.
    - Applies acronym-aware title casing.
    """
    if raw is None:
        return None
    t = raw.strip().strip("-_:;,. \t")
    # Repeatedly strip trailing volume-ish residues (could stack).
    prev = None
    while prev != t:
        prev = t
        t = _AUTHOR_TRAILING_RE.sub("", t).strip().strip("-_:;,. \t")
    t = _collapse_whitespace(t)
    if not t:
        return None
    if t.isdigit():
        return None
    if len(t) <= 2:
        return None
    if not re.search(r"[A-Za-z]", t):
        return None
    return _title_case(t)


def _normalize_title(raw: Optional[str]) -> Optional[str]:
    """Normalize a raw title string.

    - Strips leading "Vol N ", "Volume N ", "Part N " prefixes that may have
      leaked in.
    - Strips residual parens/brackets that sanitization missed.
    - Applies acronym-aware title casing.
    """
    if raw is None:
        return None
    t = raw.strip().strip("-_:;,. \t")
    if _TITLE_BARE_KEYWORD_RE.match(t):
        return None
    prev = None
    while prev != t:
        prev = t
        t = _TITLE_LEADING_RE.sub("", t).strip().strip("-_:;,. \t")
    t = _RESIDUAL_BRACKETS_RE.sub(" ", t)
    t = _collapse_whitespace(t)
    if not t:
        return None
    if _TITLE_BARE_KEYWORD_RE.match(t):
        return None
    return _title_case(t)


def _demote_title_as_author(author: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[int]]:
    """If an author captured looks like "Title Words N" (>=2 words, last is
    digit), treat it as title+chapter rather than author.

    Returns (author, title, chapter) where at most one of title/chapter may be
    non-None. If no demotion applies, returns (author, None, None).
    """
    if not author:
        return author, None, None
    parts = author.split()
    if len(parts) >= 2 and parts[-1].isdigit():
        chap = int(parts[-1])
        rest = " ".join(parts[:-1]).strip()
        return None, rest or None, chap
    return author, None, None


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


# Each pattern: (name, compiled_regex). Patterns are applied against the
# sanitized string. The regex must expose named groups: author, title, chapter.
# Subtitle is optional (captured into title).

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 1) "<Author> - <Title>; <Subtitle> - Volume <N>"
    #    (also handles ": " instead of "; ")
    (
        "author_dash_title_semi_subtitle_volume",
        re.compile(
            r"""^
            (?P<author>[^\-]+?)\s*-\s*
            (?P<title>[^;:]+?)\s*[;:]\s*(?P<subtitle>.+?)\s*
            -\s*(?:Volume|Vol|Part|Chapter|Ch|Episode|Ep)\.?\s*
            (?P<chapter>\d+)
            \s*$
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # 2) "<Author> - <Title> - Volume <N>"
    (
        "author_dash_title_dash_volume",
        re.compile(
            r"""^
            (?P<author>[^\-]+?)\s*-\s*
            (?P<title>.+?)\s*
            -\s*(?:Volume|Vol|Part|Chapter|Ch|Episode|Ep)\.?\s*
            (?P<chapter>\d+)
            \s*$
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # 3) "<Title> by <Author> - Part <N>"  (explicit Part/Volume keyword)
    (
        "title_by_author_dash_part",
        re.compile(
            r"""^
            (?P<title>.+?)\s+by\s+(?P<author>.+?)\s*
            -\s*(?:Volume|Vol|Part|Chapter|Ch|Episode|Ep)\.?\s*
            (?P<chapter>\d+)
            \s*$
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # 4) "<Title> by <Author> <N>"
    (
        "title_by_author_num",
        re.compile(
            r"""^
            (?P<title>.+?)\s+by\s+(?P<author>.+?)\s+
            (?P<chapter>\d+)
            \s*$
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # 5) "<Author> - <Title> <N>"  (trailing number, no Volume keyword)
    (
        "author_dash_title_num",
        re.compile(
            r"""^
            (?P<author>[^\-]+?)\s*-\s*
            (?P<title>.+?)\s+
            (?P<chapter>\d+)
            \s*$
            """,
            re.VERBOSE,
        ),
    ),
    # 6) "<Author>_<Title>_<N>" (underscore separated)
    (
        "author_underscore_title_underscore_num",
        re.compile(
            r"""^
            (?P<author>[^_]+)_
            (?P<title>.+)_
            (?P<chapter>\d+)
            \s*$
            """,
            re.VERBOSE,
        ),
    ),
    # 7) "<Author> - <Title>" (no chapter number)
    (
        "author_dash_title",
        re.compile(
            r"""^
            (?P<author>[^\-]+?)\s*-\s*
            (?P<title>.+?)
            \s*$
            """,
            re.VERBOSE,
        ),
    ),
    # 8) Last-resort: "<Something> <N>" (only chapter extractable)
    (
        "title_trailing_num",
        re.compile(
            r"""^
            (?P<title>.+?)\s+
            (?P<chapter>\d+)
            \s*$
            """,
            re.VERBOSE,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_caption(text: str) -> ParsedCaption:
    """Parse a Telegram media caption into structured metadata.

    Tries each pattern in :data:`PATTERNS` in order. First match wins. If no
    pattern matches, returns a ParsedCaption with confidence=0.0 and all
    fields None.
    """
    raw = text if text is not None else ""
    cleaned, removed_tokens = _sanitize(raw)

    if not cleaned:
        return ParsedCaption(
            author=None,
            title=None,
            chapter_num=None,
            confidence=0.0,
            raw=raw,
            matched_pattern=None,
        )

    for name, pattern in PATTERNS:
        m = pattern.match(cleaned)
        if not m:
            continue
        groups = m.groupdict()
        author = _clean_field(groups.get("author"))
        title = _clean_field(groups.get("title"))
        subtitle = _clean_field(groups.get("subtitle"))
        chap_raw = groups.get("chapter")
        chapter_num = int(chap_raw) if chap_raw and chap_raw.isdigit() else None

        # Reject junk single-word "authors" that are obviously not names.
        if author and name.startswith("author_") and not _looks_like_name(author):
            continue

        # Merge subtitle into title if present.
        if subtitle:
            title = f"{title} {subtitle}" if title else subtitle

        # Under the underscore pattern, replace underscores in title with spaces.
        if "_" in name and title:
            title = title.replace("_", " ")
            title = _collapse_whitespace(title)

        # --- Post-parse normalization -------------------------------------
        # 1) Clean trailing "Vol"/"Part" residues from author.
        author = _normalize_author(author)
        # 2) Clean leading "Vol N"/"Part N" residues from title.
        title = _normalize_title(title)
        # 3) If author still looks like "Title Words N" (multi-word, digit
        #    tail), demote it: move the digit to chapter_num and the rest to
        #    title. Only apply when we don't already have a chapter+title.
        if author and title is None and chapter_num is None:
            new_author, demoted_title, demoted_chap = _demote_title_as_author(author)
            if new_author is None and (demoted_title or demoted_chap is not None):
                author = None
                title = _normalize_title(demoted_title) if demoted_title else None
                chapter_num = demoted_chap

        # Confidence: 1.0 if match spans the full cleaned text AND nothing was
        # removed during sanitization. Each removed info token knocks 0.1 off.
        matched_span = m.end() - m.start()
        full_coverage = matched_span == len(cleaned)
        confidence = 1.0 if full_coverage else 0.8
        confidence -= 0.1 * len(removed_tokens)
        # Patterns without a chapter are weaker.
        if chapter_num is None:
            confidence -= 0.2
        confidence = max(0.0, min(1.0, confidence))

        return ParsedCaption(
            author=author,
            title=title,
            chapter_num=chapter_num,
            confidence=round(confidence, 3),
            raw=raw,
            matched_pattern=name,
        )

    # Nothing matched -> still try to extract a trailing number as chapter.
    trailing = re.search(r"(\d+)\s*$", cleaned)
    if trailing:
        return ParsedCaption(
            author=None,
            title=None,
            chapter_num=int(trailing.group(1)),
            confidence=0.2,
            raw=raw,
            matched_pattern=None,
        )

    return ParsedCaption(
        author=None,
        title=None,
        chapter_num=None,
        confidence=0.0,
        raw=raw,
        matched_pattern=None,
    )


def _looks_like_name(candidate: str) -> bool:
    """Heuristic: a plausible person/brand name.

    Avoids assigning obvious noise (e.g. single digits, obvious title words)
    as an author. This is intentionally lenient.
    """
    c = candidate.strip()
    if not c:
        return False
    if c.isdigit():
        return False
    # At least one letter.
    if not re.search(r"[A-Za-z]", c):
        return False
    return True
