"""Dependencias FastAPI del módulo jobs.

Singletons scope-app del scheduler, hub SSE y los dos servicios. El
registro/init se hace en ``infrastructure/lifespan.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable, Optional

from ._internal.scheduler import JobsScheduler
from ._internal.sse_hub import SseHub
from .models import LegacyJob
from .repositories.background import BackgroundJobsRepository
from .repositories.legacy import LegacyJobsRepository
from .services.background import BackgroundJobsService
from .services.legacy import LegacyJobsService

_scheduler: Optional[JobsScheduler] = None
_sse_hub: Optional[SseHub] = None
_bg_repo: Optional[BackgroundJobsRepository] = None
_bg_service: Optional[BackgroundJobsService] = None
_legacy_repo: Optional[LegacyJobsRepository] = None
_legacy_service: Optional[LegacyJobsService] = None


# ---------------------------------------------------------------------------
# Primitivas compartidas
# ---------------------------------------------------------------------------


def get_jobs_scheduler() -> JobsScheduler:
    """Singleton del scheduler. Compartido entre los dos servicios."""
    global _scheduler
    if _scheduler is None:
        _scheduler = JobsScheduler()
    return _scheduler


def get_sse_hub() -> SseHub:
    """Singleton del hub SSE."""
    global _sse_hub
    if _sse_hub is None:
        _sse_hub = SseHub()
    return _sse_hub


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------


def _get_background_jobs_repository() -> BackgroundJobsRepository:
    global _bg_repo
    if _bg_repo is None:
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


# ---------------------------------------------------------------------------
# Legacy jobs
# ---------------------------------------------------------------------------


def _get_legacy_jobs_repository() -> LegacyJobsRepository:
    global _legacy_repo
    if _legacy_repo is None:
        _legacy_repo = LegacyJobsRepository()
    return _legacy_repo


def get_legacy_jobs_service() -> LegacyJobsService:
    """Singleton del ``LegacyJobsService``."""
    global _legacy_service
    if _legacy_service is None:
        _legacy_service = LegacyJobsService(
            repo=_get_legacy_jobs_repository(),
            scheduler=get_jobs_scheduler(),
            sse_hub=get_sse_hub(),
        )
    return _legacy_service


# ---------------------------------------------------------------------------
# Dispatch table de runners legacy
# ---------------------------------------------------------------------------
# TODO(post-T22): mover esto a main.py o a un registry pattern. Mientras
# tanto, los runners (``run_chapter_detection`` etc.) viven en
# ``api/app.py`` y se importan diferidamente para evitar ciclo en import
# time (api.app importa este módulo a través del router).

JobBody = dict
JobRunner = Callable[[LegacyJob, JobBody], Awaitable[None]]


def get_legacy_jobs_dispatch_table() -> dict[str, JobRunner]:
    """Mapa ``job_type → runner_callable`` para los endpoints legacy.

    Late import a ``api.app`` por orden de inicialización (api.app
    importa este módulo a través del router de jobs).
    """
    from api import app as _app

    async def _chapters(job: LegacyJob, body: JobBody) -> None:
        await _app.run_chapter_detection(job)

    async def _subtitles(job: LegacyJob, body: JobBody) -> None:
        await _app.run_subtitle_generation(job)

    async def _translate(job: LegacyJob, body: JobBody) -> None:
        await _app.run_translation(job)

    async def _dubbing(job: LegacyJob, body: JobBody) -> None:
        await _app.run_dubbing(
            job,
            body.get("voice_profile"),
            body.get("use_model_voice", False),
        )

    return {
        "chapters": _chapters,
        "subtitles": _subtitles,
        "translate": _translate,
        "dubbing": _dubbing,
    }


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def reset_for_tests() -> None:
    """Limpia los singletons entre tests."""
    global _scheduler, _sse_hub, _bg_repo, _bg_service, _legacy_repo, _legacy_service
    _scheduler = None
    _sse_hub = None
    _bg_repo = None
    _bg_service = None
    _legacy_repo = None
    _legacy_service = None
