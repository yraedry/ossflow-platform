"""SRT subtitle file writer with punctuation-aware line breaking."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import SubtitleConfig
from .timestamp_fixer import TimestampFixer
from .utils import format_timestamp

log = logging.getLogger("subtitler")

# Break priorities (lower = preferred break point)
_SENTENCE_END = re.compile(r"[.!?]$")
_CLAUSE_END = re.compile(r"[,:;]$")
_CONJUNCTIONS = {"and", "but", "or", "so", "yet", "nor", "for", "because",
                 "although", "while", "when", "then", "also"}


class SubtitleWriter:
    """Generate SRT files with punctuation-aware line breaking and two-line support."""

    def __init__(self, config: SubtitleConfig, timestamp_fixer: TimestampFixer) -> None:
        self.config = config
        self.fixer = timestamp_fixer

    def write_srt(self, words: list[dict], output_path: Path) -> list[dict]:
        """Build subtitle blocks from words and write to SRT file.

        Returns the list of subtitle dicts for downstream validation.
        """
        words = self.fixer.fix_words(words)

        # Build raw subtitle blocks
        subtitles = self._build_blocks(words)

        # Fix timing at subtitle level
        subtitles = self.fixer.fix_subtitles(subtitles)

        # Write SRT
        self._write_file(subtitles, output_path)

        return subtitles

    def _build_blocks(self, words: list[dict]) -> list[dict]:
        """Group words into subtitle blocks respecting character AND duration limits."""
        max_chars = self.config.max_chars_per_line
        max_lines = self.config.max_lines
        max_total = max_chars * max_lines  # Total chars allowed across all lines
        max_dur = self.config.max_duration

        subtitles: list[dict] = []
        buf_words: list[dict] = []
        buf_len = 0  # Track length of text that would be produced (words + spaces)

        for word_data in words:
            word = word_data.get("word", "").strip()
            if not word:
                continue

            addition = len(word) + (1 if buf_words else 0)  # +1 for space if not first

            # Check duration limit: would adding this word exceed max_duration?
            exceeds_duration = False
            if buf_words and max_dur > 0:
                buf_start = buf_words[0].get("start", 0.0)
                word_end = word_data.get("end", word_data.get("start", 0.0))
                if word_end - buf_start > max_dur:
                    exceeds_duration = True

            if (buf_len + addition > max_total or exceeds_duration) and buf_words:
                # Flush buffer
                subtitles.append(self._flush_block(buf_words))
                buf_words = [word_data]
                buf_len = len(word)
            else:
                buf_words.append(word_data)
                buf_len += addition

        if buf_words:
            subtitles.append(self._flush_block(buf_words))

        return subtitles

    def _flush_block(self, buf_words: list[dict]) -> dict:
        """Create a subtitle dict from buffered words."""
        text = " ".join(w.get("word", "").strip() for w in buf_words)
        start = buf_words[0].get("start", 0.0)
        end = buf_words[-1].get("end", start + 0.5)
        formatted = self._format_lines(text)
        return {"start": start, "end": end, "text": formatted}

    def _format_lines(self, text: str) -> str:
        """Split text into up to 2 lines, preferring punctuation-aware break points."""
        max_chars = self.config.max_chars_per_line

        if len(text) <= max_chars:
            return text

        words = text.split()
        if len(words) <= 1:
            return text

        best_break = self._find_best_break(words, max_chars)

        line1 = " ".join(words[:best_break])
        line2 = " ".join(words[best_break:])

        # If either line still exceeds limit, truncate at last complete word.
        # rsplit(" ", 1)[0] sobre un string SIN espacios devuelve "" — una
        # palabra larga sin espacios (compuestos o URLs) se perdería entera.
        # _truncate_at_word usa split en espacios y va quitando palabras del
        # final hasta caber; si al final no queda nada, hace fallback a un
        # hard cut para al menos mostrar caracteres en lugar de la cadena
        # vacía silenciosa que borraba el subtítulo.
        line1 = _truncate_at_word(line1, max_chars)
        line2 = _truncate_at_word(line2, max_chars)

        return f"{line1}\n{line2}"

    def _find_best_break(self, words: list[str], max_chars: int) -> int:
        """Find the best word index to break at, preferring punctuation and balanced lines."""
        n = len(words)
        target = len(" ".join(words)) / 2  # Aim for balanced lines

        # Score each possible break point
        best_idx = n // 2
        best_score = float("inf")

        for i in range(1, n):
            line1 = " ".join(words[:i])
            line2 = " ".join(words[i:])

            # Both lines must fit
            if len(line1) > max_chars or len(line2) > max_chars:
                # Still consider if nothing else works
                penalty = 1000
            else:
                penalty = 0

            # Balance score: how far from center
            balance = abs(len(line1) - target)

            # Punctuation bonus (lower is better)
            prev_word = words[i - 1]
            punct_bonus = 30  # Default: no punctuation benefit
            if _SENTENCE_END.search(prev_word):
                punct_bonus = 0
            elif _CLAUSE_END.search(prev_word):
                punct_bonus = 10
            elif words[i].lower() in _CONJUNCTIONS:
                # Break before conjunction
                punct_bonus = 20

            score = penalty + balance + punct_bonus
            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _write_file(self, subtitles: list[dict], output_path: Path) -> None:
        """Write subtitle blocks to SRT file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, sub in enumerate(subtitles, 1):
                start_ts = format_timestamp(sub["start"])
                end_ts = format_timestamp(sub["end"])
                f.write(f"{idx}\n{start_ts} --> {end_ts}\n{sub['text']}\n\n")
        log.info("Wrote %d subtitle blocks to %s", len(subtitles), output_path.name)


def _truncate_at_word(line: str, max_chars: int) -> str:
    """Truncate ``line`` to ``max_chars`` at a word boundary.

    Fallback robusto frente al bug clásico ``s[:n].rsplit(" ",1)[0]`` que
    devolvía "" cuando la porción truncada no contenía ningún espacio
    (palabra única demasiado larga). Ahora:

    * Si la línea ya entra, se devuelve tal cual.
    * Si hay espacios, se eliminan palabras enteras del final hasta caber.
    * Si no queda nada (toda la línea es una única palabra muy larga),
      se aplica un hard cut a ``max_chars`` en vez de devolver "" — mejor
      un subtítulo truncado que un subtítulo vacío.
    """
    if len(line) <= max_chars:
        return line
    words = line.split()
    while words and len(" ".join(words)) > max_chars:
        words.pop()
    if words:
        return " ".join(words)
    return line[:max_chars]
