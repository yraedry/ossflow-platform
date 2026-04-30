"""DTOs Pydantic v2 del módulo jobs.

Espejo de las dataclasses de dominio con validación HTTP. Mantienen los
nombres de campo del JSON externo heredado para no romper el frontend.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class BackgroundJobResponse(BaseModel):
    """Forma del payload de ``GET /api/background-jobs/{id}``."""

    id: str
    type: str
    status: str
    progress: Optional[float] = None
    message: str = ""
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    params: dict[str, Any] = {}


class BackgroundJobListResponse(BaseModel):
    """Forma del payload de ``GET /api/background-jobs``."""

    jobs: list[BackgroundJobResponse]
