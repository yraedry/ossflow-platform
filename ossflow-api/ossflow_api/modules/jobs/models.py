"""Entidades de dominio del módulo jobs.

Dataclasses puras: cero acoplamiento a SQLAlchemy, FastAPI o asyncio.
Los nombres de campo son distintos a propósito entre ``BackgroundJob`` y
``LegacyJob`` (``id`` vs ``job_id``, ``type`` vs ``job_type``) para
preservar el JSON shape externo de los endpoints heredados.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    """Estados posibles de un job.

    Compartido por ``BackgroundJob`` y ``LegacyJob`` — los strings son
    idénticos en ambos sistemas heredados.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackgroundJob:
    """Job persistido en tabla ``background_jobs`` (DB-backed, polling).

    Consumido por ``cleanup``, ``duplicates`` y ``burn_subs`` (que se
    elimina en T22). ``params`` es un diccionario opaco con los argumentos
    del runner (e.g. ``{"path": "/media/X"}``).
    """

    id: str
    type: str
    status: str = JobStatus.QUEUED.value
    progress: Optional[float] = None
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LegacyJob:
    """Job persistido en tabla ``legacy_jobs`` (DB-backed, SSE).

    Consumido por ``elevenlabs`` y los endpoints chapter/subs/dub legacy
    de ``app.py`` hasta que se migren en F5/T20-T22. El campo
    ``video_path`` es dominio puro y se conserva tipado para no romper el
    JSON shape del frontend.
    """

    job_id: str
    job_type: str
    video_path: str
    status: str = JobStatus.QUEUED.value
    progress: Optional[float] = None
    message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    result: Optional[dict] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
