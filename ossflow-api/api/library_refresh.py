"""Compat shim. Lógica movida a ``ossflow_api.modules.library.refresh``.

Mantiene la API legacy completa — públicas y privadas — porque varios
consumidores (incluyendo ``pipeline.py`` y muchos tests) hacen
monkeypatch sobre símbolos privados como ``_probe_track_languages``.

Cuando esos consumidores migren a importar desde
``ossflow_api.modules.library``, este shim se elimina (F5 / T_LATE).
"""

from ossflow_api.modules.library.refresh import (  # noqa: F401
    VIDEO_EXTENSIONS,
    _CHAPTER_RE,
    _DUB_SUFFIXES,
    _ENGLISH_LANG_TAGS,
    _SPANISH_LANG_TAGS,
    _file_fingerprint,
    _has_english_subtitle,
    _has_spanish_audio,
    _has_spanish_subtitle,
    _probe_duration,
    _probe_track_languages,
    _video_flags,
    ensure_duration,
    rediscover_instructional,
    refresh_instructional_flags,
    refresh_many,
)
