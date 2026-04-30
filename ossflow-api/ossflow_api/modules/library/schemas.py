"""DTOs Pydantic del módulo library.

Mantiene los shapes JSON exactos que el frontend espera (ver
``frontend/src/api/library.js`` y consumidores). Migrar a Pydantic
permite validación de inputs y documentación automática en /docs sin
cambiar el contrato.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ScanRequest(BaseModel):
    """Body de ``POST /api/scan``."""

    path: str = Field(default="", description="Library root. Si vacío, usa settings.library_path.")
