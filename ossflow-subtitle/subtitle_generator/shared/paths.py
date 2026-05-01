"""Helpers puros de paths SRT del servicio subtitle-generator.

Migrados de ``app.py`` en T31.1. Funciones puras sin dependencias
externas; usadas por translate_runner, analyzer y otros componentes
de ``core/``.
"""

from __future__ import annotations

from pathlib import Path


def resolve_input(path: Path) -> Path:
    """Valida que el input path exista. Devuelve el path (file o directory)."""
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def clean_base_stem(srt_path: Path) -> str:
    """Devuelve el stem base de un SRT, eliminando tags de idioma.

    Ejemplos:
      "S01E01 foo.en.srt"  -> "S01E01 foo"
      "S01E01 foo_EN.srt"  -> "S01E01 foo"
      "S01E01 foo.srt"     -> "S01E01 foo"
    """
    stem = srt_path.stem
    for tag in (".en", ".EN", "_en", "_EN"):
        if stem.endswith(tag):
            return stem[: -len(tag)]
    return stem


def literal_srt_path_for(srt_path: Path) -> Path:
    """Devuelve el sidecar ``<video-stem>.es.srt`` (subtítulo literal)."""
    return srt_path.with_name(f"{clean_base_stem(srt_path)}.es.srt")


def dub_srt_path_for(srt_path: Path) -> Path:
    """Devuelve ``<video-stem>.dub.es.srt`` (script iso-sync para doblaje)."""
    return srt_path.with_name(f"{clean_base_stem(srt_path)}.dub.es.srt")


def words_json_for(srt_path: Path) -> Path:
    """Devuelve ``<video>.words.json`` adyacente al SRT.

    El SRT puede ser ``.en.srt``, ``.srt`` o ya ``.es.srt``: siempre
    se vuelve al stem base del vídeo.
    """
    return srt_path.with_name(f"{clean_base_stem(srt_path)}.words.json")
