"""Constantes y patrones del módulo chapters.

No hay DTOs Pydantic porque los endpoints aceptan JSON ad-hoc; agrupamos
aquí los regex y la lista de sidecars para que router/service/repository
compartan exactamente las mismas reglas.
"""

from __future__ import annotations

import re

# Regex para parsear ``{prefix} - SNNeMM - {title}{ext}`` en el nombre del
# fichero (no en el path completo).
# Ej.: "John Danaher - S01E03 - Armbar Fundamentals.mkv"
#      prefix = "John Danaher", season=01, ep=03, ext=".mkv"
SNNEMM_RE = re.compile(
    r"^(?P<prefix>.*?)\s*-\s*S(?P<season>\d{2})E(?P<ep>\d{2,3})\s*-\s*.*(?P<ext>\.[^.]+)$"
)

# Formato alternativo del splitter sin renombrar: "1-2.mp4" → vol=1, ep=2.
RAW_RE = re.compile(r"^(?P<vol>\d+)-(?P<ep>\d+)(?P<ext>\.[^.]+)$")

# Caracteres ilegales en filenames de Windows (y que no queremos en ningún OS).
ILLEGAL_RE = re.compile(r'[\/\\:*?"<>|]')
WS_RE = re.compile(r"\s+")

# Sufijos de archivos hermanos que renombramos junto al vídeo principal.
# Cada entrada reemplaza la extensión completa del vídeo
# (de "Name.mkv" → "Name.srt" / "Name.en.srt" / ...).
SIDECAR_SUFFIXES: tuple[str, ...] = (
    ".srt",
    ".en.srt",
    ".es.srt",
    ".ES.srt",
    "_ESP_DUB.srt",
    "_DOBLADO.mkv",
    "_DOBLADO.mp4",
)

# Extensiones de vídeo que escaneamos en ``rename-by-oracle``.
VIDEO_EXTS: frozenset[str] = frozenset({".mkv", ".mp4", ".avi", ".mov"})

# Longitud máxima del título saneado (antes era hard-coded en _sanitize_title).
MAX_TITLE_LEN = 120
