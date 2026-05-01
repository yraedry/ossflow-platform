"""Helpers de parsing SRT + prosodic continuity.

Migrado de ``pipeline.py`` en T32.1. Funciones puras sin estado.
``SrtBlock`` se reexporta desde ``sync.aligner`` (donde está definido)
para mantener un único source of truth.
"""

from __future__ import annotations

import re
from pathlib import Path

from dubbing_generator.sync.aligner import SrtBlock


def parse_time(time_str: str) -> int:
    """Parsea ``HH:MM:SS,mmm`` a milisegundos."""
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def parse_srt(srt_path: Path) -> list[SrtBlock]:
    """Parsea un fichero SRT en una lista de :class:`SrtBlock`."""
    content = srt_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\d+)\n"
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n"
        r"(.*?)(?=\n\n|\n$|\Z)",
        re.DOTALL,
    )
    blocks: list[SrtBlock] = []
    for m in pattern.finditer(content):
        text = m.group(4).replace("\n", " ").strip()
        text = re.sub(r"\((.*?)\)", r"\1", text)
        blocks.append(SrtBlock(
            index=int(m.group(1)),
            start_ms=parse_time(m.group(2)),
            end_ms=parse_time(m.group(3)),
            text=text,
        ))
    return blocks


_CONTINUATION_STARTS = (
    "y ", "o ", "u ", "e ", "pero ", "porque ", "pues ", "así que ",
    "aunque ", "sino ", "mientras ", "cuando ", "donde ", "como ",
    "que ", "para ", "al ", "del ", "de ", "en ", "con ", "sin ",
    "sobre ", "entre ", "hasta ", "desde ", "a ",
)


def apply_prosodic_continuity(text: str, next_text: str | None) -> str:
    """Ajusta puntuación final para que el TTS no cierre prosodia.

    Si la siguiente frase continúa el discurso (empieza con minúscula
    o con conector), cambia el punto final ``.`` por coma para que el
    TTS no marque cierre entonativo. Signos fuertes (``!`` ``?``)
    quedan intactos porque sí marcan intención. Sin siguiente frase,
    deja el punto (es el cierre real del bloque).
    """
    if not text or not next_text:
        return text
    stripped = text.rstrip()
    if not stripped.endswith("."):
        return text
    if stripped.endswith("...") or stripped.endswith(".."):
        return text

    nxt_clean = next_text.lstrip()
    if not nxt_clean:
        return text

    first_char = nxt_clean[0]
    continues = first_char.islower()
    if not continues:
        lower_nxt = nxt_clean.lower()
        continues = any(lower_nxt.startswith(c) for c in _CONTINUATION_STARTS)

    if not continues:
        return text

    trailing_ws = text[len(stripped):]
    return stripped[:-1] + "," + trailing_ws
