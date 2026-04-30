"""DTOs Pydantic del módulo subtitles.

Replican el contrato HTTP del antiguo ``api/subtitles.py`` para preservar
la compatibilidad con el frontend y los tests existentes.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ValidateBody(BaseModel):
    srt_path: str


class RegenerateBody(BaseModel):
    srt_path: str
    segment_idx: int
    context_seconds: float = 1.0
    video_path: Optional[str] = None
    model: str = "large-v3"
    language: str = "en"


class ApplyBody(BaseModel):
    srt_path: str
    segment_idx: int
    text: str
    start: Optional[float] = None
    end: Optional[float] = None


class TranslateBody(BaseModel):
    srt_path: str
    target_lang: str = "ES"
    source_lang: str = "EN"
    provider: Optional[str] = None
    model: Optional[str] = None
    formality: Optional[str] = None
    api_key: Optional[str] = None
    fallback_provider: Optional[str] = None
    fallback_api_key: Optional[str] = None
    out_path: Optional[str] = None
    dubbing_mode: bool = False
    dubbing_cps: Optional[float] = None


class AnalyzeBody(BaseModel):
    video_path: str
    language: str = "en"
    model: str = "large-v3"
