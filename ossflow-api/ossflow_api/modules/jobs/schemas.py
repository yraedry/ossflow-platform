"""DTOs Pydantic v2 del módulo jobs.

Espejo de las dataclasses de dominio con validación HTTP. Mantienen los
nombres de campo del JSON externo heredado para no romper el frontend.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Background jobs (/api/background-jobs/*)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Legacy jobs (/api/jobs/*)
# ---------------------------------------------------------------------------


class LegacyJobCreateRequest(BaseModel):
    """Payload de ``POST /api/jobs``.

    Campos según el contrato actual de ``app.py:api_create_job``: tipo de
    job + path del vídeo + opciones específicas del runner. Las opciones
    se pasan ``extra='allow'`` para no romper extensiones futuras del
    frontend (e.g. ``voice_profile``, ``use_model_voice``).
    """

    model_config = {"extra": "allow"}

    type: str
    path: str


class LegacyJobCreateResponse(BaseModel):
    """Payload de respuesta de ``POST /api/jobs``."""

    job_id: str
    status: str


class LegacyJobResponse(BaseModel):
    """Forma del payload de ``GET /api/jobs/{id}``."""

    job_id: str
    job_type: str
    video_path: str
    status: str
    progress: Optional[float] = None
    message: str = ""
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[dict[str, Any]] = None


class LegacyJobListResponse(BaseModel):
    """Forma del payload de ``GET /api/jobs``."""

    jobs: list[LegacyJobResponse]
