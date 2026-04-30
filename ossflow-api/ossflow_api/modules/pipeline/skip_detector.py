"""Detección de "ya hecho" para saltar pasos del pipeline.

Migrado de ``api/pipeline.py`` en T_LATE_2.3. Cada función responde
si un capítulo (o toda una Season) ya tiene el artefacto que el
step iba a producir, para evitar re-procesar.

Sources of truth (orden de checks dentro de cada función):

* sidecars: ``<base>.en.srt``, ``<base>.es.srt``, ``<base>.dub.es.srt``,
  ``<base>_DOBLADO.{mkv,mp4}``, ``doblajes/<base>.mkv``,
  ``elevenlabs/<chapter.name>``.
* ffprobe sobre el ``.mkv`` post-promote: detecta tracks de audio /
  subtítulos por código de idioma. ``_probe_track_languages`` vive en
  ``ossflow_api.modules.library.refresh``; los tests parchean por
  esa ruta absoluta.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")
DUB_SUFFIXES = ("_DOBLADO.mkv", "_DOBLADO.mp4")
CHAPTER_SE_RE = re.compile(r"S\d{2}E\d{2}", re.IGNORECASE)


def chapter_has_en_subs(chapter: Path) -> bool:
    """True si el capítulo ya tiene subtítulos en inglés.

    Checks: ``<base>.en.srt`` / ``<base>.srt`` sidecar, o un track
    de subtítulos embebido tagged eng/en/english.
    """
    folder = chapter.parent
    base = chapter.stem
    if (folder / f"{base}.en.srt").exists() or (folder / f"{base}.srt").exists():
        return True
    try:
        from ossflow_api.modules.library.refresh import (
            _has_english_subtitle,
            _probe_track_languages,
        )
        _, sub_langs = _probe_track_languages(chapter)
        if _has_english_subtitle(sub_langs):
            return True
    except Exception as exc:
        log.debug("ffprobe failed for %s: %s", chapter, exc)
    return False


def chapter_has_es_subs(chapter: Path, dubbing_mode: bool) -> bool:
    """True si el capítulo ya tiene subtítulos en español del track pedido.

    En ``dubbing_mode``, el step translate produce ``<base>.dub.es.srt``
    (anchor-aware). En modo normal produce ``<base>.es.srt`` literal.
    Subs ES embebidos satisfacen cualquiera (solo aparecen post-promote
    y siempre llevan el track literal).
    """
    folder = chapter.parent
    base = chapter.stem
    if dubbing_mode:
        if (folder / f"{base}.dub.es.srt").exists():
            return True
    else:
        candidates = (
            f"{base}.es.srt",
            f"{base}.ES.srt",
            f"{base}_ES.srt",
            f"{base}_ESP_DUB.srt",
        )
        if any((folder / c).exists() for c in candidates):
            return True
    try:
        from ossflow_api.modules.library.refresh import (
            _has_spanish_subtitle,
            _probe_track_languages,
        )
        _, sub_langs = _probe_track_languages(chapter)
        if _has_spanish_subtitle(sub_langs):
            return True
    except Exception as exc:
        log.debug("ffprobe failed for %s: %s", chapter, exc)
    return False


def chapter_is_dubbed(chapter: Path) -> bool:
    """True si el capítulo ya tiene doblaje en español.

    Sources of truth (cualquiera basta):
    * legacy XTTS:               ``<base>_DOBLADO.{mkv,mp4}`` adyacente.
    * flujo B v5 pre-promote:    ``<folder>/doblajes/<base>.mkv``.
    * Studio E2E pre-promote:    ``<folder>/elevenlabs/<chapter.name>``.
    * promoted multi-track:      audio stream tagged spa/es/spanish.

    El cuarto check es crítico: tras ``promote``, el ``.mkv`` lleva el
    audio doblado como segundo stream y ``doblajes/`` se borra. Sin
    ffprobe, una segunda corrida del step dubbing re-doblaría todo.
    """
    folder = chapter.parent
    base = chapter.stem
    if any((folder / f"{base}{sfx}").exists() for sfx in DUB_SUFFIXES):
        return True
    if (folder / "doblajes" / f"{base}.mkv").exists():
        return True
    if (folder / "elevenlabs" / chapter.name).exists():
        return True
    try:
        from ossflow_api.modules.library.refresh import (
            _has_spanish_audio,
            _probe_track_languages,
        )
        audio_langs, _ = _probe_track_languages(chapter)
        if _has_spanish_audio(audio_langs):
            return True
    except Exception as exc:
        log.debug("ffprobe failed for %s: %s", chapter, exc)
    return False


def list_chapters(season_dir: Path) -> list[Path]:
    if not season_dir.exists() or not season_dir.is_dir():
        return []
    return [
        p for p in season_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTS
        and CHAPTER_SE_RE.search(p.name)
    ]


def season_already_dubbed(season_dir: Path) -> tuple[bool, int, int]:
    """Devuelve ``(all_dubbed, dubbed_count, total_count)``.

    Seasons vacías → ``(False, 0, 0)`` (no hay nada que skipear).
    """
    chapters = list_chapters(season_dir)
    if not chapters:
        return False, 0, 0
    dubbed = sum(1 for c in chapters if chapter_is_dubbed(c))
    return dubbed == len(chapters), dubbed, len(chapters)


def season_already_subbed_en(season_dir: Path) -> tuple[bool, int, int]:
    chapters = list_chapters(season_dir)
    if not chapters:
        return False, 0, 0
    n = sum(1 for c in chapters if chapter_has_en_subs(c))
    return n == len(chapters), n, len(chapters)


def season_already_subbed_es(
    season_dir: Path, dubbing_mode: bool,
) -> tuple[bool, int, int]:
    chapters = list_chapters(season_dir)
    if not chapters:
        return False, 0, 0
    n = sum(1 for c in chapters if chapter_has_es_subs(c, dubbing_mode))
    return n == len(chapters), n, len(chapters)
