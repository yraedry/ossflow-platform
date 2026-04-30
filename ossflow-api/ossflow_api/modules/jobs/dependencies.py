"""Dependencias FastAPI del módulo jobs.

Singletons scope-app del scheduler y los servicios. El registro/init se
hace en ``infrastructure/lifespan.py`` (T19.4 / T19.6).

Por ahora solo se compone la parte ``background``. ``LegacyJobsService``
se añade en T19.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._internal.scheduler import JobsScheduler
from ._internal.sse_hub import SseHub
from .repositories.background import BackgroundJobsRepository
from .services.background import BackgroundJobsService

_scheduler: Optional[JobsScheduler] = None
_sse_hub: Optional[SseHub] = None
_bg_repo: Optional[BackgroundJobsRepository] = None
_bg_service: Optional[BackgroundJobsService] = None


def get_jobs_scheduler() -> JobsScheduler:
    """Singleton del scheduler. Compartido entre los dos servicios."""
    global _scheduler
    if _scheduler is None:
        _scheduler = JobsScheduler()
    return _scheduler


def get_sse_hub() -> SseHub:
    """Singleton del hub SSE. Solo lo usa ``LegacyJobsService`` (T19.5).

    Se expone aquí para tests que quieren mockearlo, y para que el
    lifespan pueda limpiar las colas en shutdown.
    """
    global _sse_hub
    if _sse_hub is None:
        _sse_hub = SseHub()
    return _sse_hub


def _get_background_jobs_repository() -> BackgroundJobsRepository:
    global _bg_repo
    if _bg_repo is None:
        # Import diferido a la infraestructura: ``CONFIG_DIR`` se resuelve
        # contra env vars al primer uso, no al import time.
        from ossflow_api.infrastructure.config import CONFIG_DIR
        history = Path(CONFIG_DIR) / "background_jobs.json"
        _bg_repo = BackgroundJobsRepository(history_file=history)
    return _bg_repo


def get_background_jobs_service() -> BackgroundJobsService:
    """Singleton del ``BackgroundJobsService``."""
    global _bg_service
    if _bg_service is None:
        _bg_service = BackgroundJobsService(
            repo=_get_background_jobs_repository(),
            scheduler=get_jobs_scheduler(),
        )
    return _bg_service


def reset_for_tests() -> None:
    """Limpia los singletons entre tests."""
    global _scheduler, _sse_hub, _bg_repo, _bg_service
    _scheduler = None
    _sse_hub = None
    _bg_repo = None
    _bg_service = None
