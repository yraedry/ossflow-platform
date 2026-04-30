"""DTOs Pydantic del módulo promote.

Replican el contrato HTTP del antiguo ``api/promote.py`` para no romper
el frontend ni los consumidores existentes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class PromoteChapterBody(BaseModel):
    """Payload de ``POST /api/promote/chapter``."""

    video_path: str  # ruta host del ORIGINAL (p.ej. <Season>/<name>.mp4)


class PromoteSeasonBody(BaseModel):
    """Payload de ``POST /api/promote/season``."""

    season_path: str


@dataclass
class Inputs:
    """Rutas resueltas para promocionar un único capítulo.

    Se exporta como ``Inputs`` (con alias ``_Inputs`` retrocompatible para
    los tests antiguos) para que el servicio y los helpers compartan la
    misma estructura sin acoplarse al router.
    """

    original: Path                   # <Season>/<name>.mp4 (o .mkv, etc.)
    dubbed: Path                     # <Season>/doblajes/<name>.mkv
    output: Path                     # <Season>/<name>.mkv (final)
    output_tmp: Path                 # <Season>/<name>.mkv.tmp
    es_srt: Optional[Path]
    en_srt: Optional[Path]
    sidecars_to_delete: list[Path]


# Alias retrocompatible: la implementación previa usaba ``_Inputs``.
_Inputs = Inputs
