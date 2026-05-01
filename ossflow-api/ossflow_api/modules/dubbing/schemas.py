"""DTOs Pydantic del módulo dubbing.

Replican el contrato HTTP del antiguo ``api/dubbing.py`` para no
romper el frontend ni los consumidores existentes.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class AnalyzeBody(BaseModel):
    """Payload de ``POST /api/dubbing/analyze``."""

    video_path: str
    srt_path: Optional[str] = None
    synthesize: bool = False
    max_phrases: Optional[int] = None
    voice_profile: Optional[str] = None


class VoiceTranscriptBody(BaseModel):
    """Payload de ``PUT /api/dubbing/voices/{filename}/transcript``."""

    transcript: str
