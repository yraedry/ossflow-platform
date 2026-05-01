"""Singleton ``SegmentRegenerator`` cacheado por (model, language).

Migrado de ``app.py`` en T31.2. Sustituye al global module-level
``_regenerator`` por una caché ``functools.lru_cache`` que reusa la
instancia mientras el (model, language) sea el mismo. El cache se
puede invalidar para tests vía ``get_regenerator.cache_clear()``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=4)
def get_regenerator(model: str, language: str) -> Any:
    """Devuelve un ``SegmentRegenerator`` cacheado por ``(model, language)``.

    Carga perezosa de las dependencias pesadas para evitar el coste en
    arranque cuando el endpoint /regenerate-segment no se usa.
    """
    from subtitle_generator.config import TranscriptionConfig
    from subtitle_generator.segment_regen import SegmentRegenerator

    return SegmentRegenerator(
        TranscriptionConfig(model_name=model, language=language),
    )
