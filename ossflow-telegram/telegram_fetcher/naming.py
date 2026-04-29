"""Filesystem-safe naming helpers.

Contract (architecture section 7):
    folder   = sanitize("<Author> - <Title>")
    filename = sanitize("<Author> - <Title><N>.<ext>")  (no space before N)

The stem of the final filename MUST satisfy:
    re.search(r"(\\d+)\\s*$", stem).group(1) == str(chapter_num)
so that the oracle splitter picks up Volume N correctly.
"""
from __future__ import annotations

import re

_FORBIDDEN_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTI_WS_RE = re.compile(r"\s+")
_RESERVED_NAMES = {
    "CON", "PRN", "NUL", "AUX",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}

MAX_SEGMENT_LEN = 200


def sanitize_for_windows(s: str) -> str:
    """Strip forbidden characters, normalize whitespace, enforce Windows rules."""
    if s is None:
        return ""
    # Collapse whitespace (incl. tabs/newlines) first so they don't become '_'.
    out = _MULTI_WS_RE.sub(" ", s).strip()
    out = _FORBIDDEN_RE.sub("_", out)
    out = _MULTI_WS_RE.sub(" ", out).strip()
    # Strip trailing dots/spaces (invalid on Windows).
    out = out.rstrip(". ")
    if not out:
        return "_"
    # Reserved device names (case-insensitive, with or without extension).
    stem = out.split(".", 1)[0].upper()
    if stem in _RESERVED_NAMES:
        out = "_" + out
    if len(out) > MAX_SEGMENT_LEN:
        out = out[:MAX_SEGMENT_LEN].rstrip(". ")
    return out


def instructional_dirname(author: str, title: str) -> str:
    """Return folder name for an instructional: '<Author> - <Title>'."""
    base = f"{(author or '').strip()} - {(title or '').strip()}"
    return sanitize_for_windows(base)


def chapter_filename(author: str, title: str, chapter_num: int, ext: str) -> str:
    """Return filename for a single chapter: '<Author> - <Title><N>.<ext>' (no space before N)."""
    if not ext.startswith("."):
        ext = "." + ext
    # Sanitize the stem first; then append N and ext.
    stem = f"{(author or '').strip()} - {(title or '').strip()}"
    stem = sanitize_for_windows(stem)
    # Add chapter number directly (no space). Keep room under MAX_SEGMENT_LEN.
    suffix = f"{int(chapter_num)}{ext}"
    if len(stem) + len(suffix) > MAX_SEGMENT_LEN:
        stem = stem[: MAX_SEGMENT_LEN - len(suffix)].rstrip(". ")
    return f"{stem}{suffix}"
