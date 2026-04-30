"""Servicio público de "background jobs".

POPO. Recibe ``BackgroundJobsRepository`` y ``JobsScheduler`` por
constructor — testeable sin TestClient ni mocks de FastAPI.

Sustituye al antiguo ``api.background_jobs.JobRegistry``. La API es
compatible: ``submit(type, coro_factory, params)`` + ``get`` + ``list_all``
+ ``init``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Awaitable, Callable, Optional

from ..models import BackgroundJob, JobStatus
from .._internal.scheduler import JobsScheduler
from ..repositories.background import BackgroundJobsRepository, MAX_ENTRIES

log = logging.getLogger(__name__)

ProgressCallback = Callable[[Optional[float], str], None]
CoroFactory = Callable[[ProgressCallback], Awaitable[dict]]


class BackgroundJobsService:
    """Encola y gestiona el ciclo de vida de los background jobs.

    El scheduling usa ``run_detached`` (thread propio + ``asyncio.run``)
    porque los jobs deben sobrevivir al ciclo de la request HTTP que los
    lanzó (clave en ``cleanup_scan`` que dura minutos, y para que
    ``TestClient`` no destruya el loop antes de tiempo).
    """

    def __init__(
        self,
        repo: BackgroundJobsRepository,
        scheduler: JobsScheduler,
    ) -> None:
        self._repo = repo
        self._scheduler = scheduler
        self._initialized = False

    # --- arranque ---------------------------------------------------------

    def init(self) -> None:
        """Hook de startup: crea tablas, importa legacy, recupera huérfanos.

        Idempotente — invocaciones posteriores son no-op.
        """
        if self._initialized:
            return
        recovered = self._repo.init_db_and_recover()
        if recovered:
            log.info("Background jobs: %d huérfanos marcados como FAILED", recovered)
        self._initialized = True

    # --- consultas -------------------------------------------------------

    def get(self, job_id: str) -> Optional[BackgroundJob]:
        return self._repo.get(job_id)

    def list_all(self, type_filter: Optional[str] = None) -> list[BackgroundJob]:
        return self._repo.list_all(type_filter=type_filter)

    # --- submit ----------------------------------------------------------

    def submit(
        self,
        type: str,
        coro_factory: CoroFactory,
        params: Optional[dict] = None,
    ) -> BackgroundJob:
        """Crea el job, lo persiste y lo lanza en background.

        ``coro_factory`` recibe un callback ``update_progress(percent,
        message)`` y devuelve un ``dict`` que se almacena en ``job.result``.
        Cualquier excepción → ``status=FAILED`` con ``error`` poblado.
        """
        job_id = uuid.uuid4().hex[:12]
        job = BackgroundJob(
            id=job_id,
            type=type,
            params=dict(params or {}),
        )

        # Persiste en estado QUEUED antes de lanzar el thread (el listado
        # debe ver el job aunque el scheduler tarde un tick en arrancarlo).
        self._repo.upsert(job)
        self._repo.trim_to(MAX_ENTRIES)

        def update_progress(percent: Optional[float], message: str = "") -> None:
            if percent is not None:
                try:
                    job.progress = float(percent)
                except (TypeError, ValueError):
                    pass
            if message:
                job.message = message
            # No persistimos cada update_progress — sería costoso. Solo en
            # transiciones de status. El estado en memoria del job es
            # suficiente para que el listado refleje el progreso.

        async def _runner() -> None:
            job.status = JobStatus.RUNNING.value
            self._repo.upsert(job)
            try:
                result = await coro_factory(update_progress)
                job.result = result if isinstance(result, dict) else {"value": result}
                job.status = JobStatus.COMPLETED.value
                job.progress = 100.0
            except Exception as exc:  # noqa: BLE001 — todos los errores → FAILED
                log.exception("Background job %s failed", job_id)
                job.status = JobStatus.FAILED.value
                job.error = f"{exc.__class__.__name__}: {exc}"
            finally:
                job.completed_at = datetime.now().isoformat()
                self._repo.upsert(job)
                self._repo.trim_to(MAX_ENTRIES)

        try:
            self._scheduler.run_detached(lambda: _runner(), name=f"bgjob-{job_id}")
        except RuntimeError as exc:
            # Fallo de scheduling: marca FAILED y persiste el cambio.
            job.status = JobStatus.FAILED.value
            job.error = f"scheduling failed: {exc}"
            job.completed_at = datetime.now().isoformat()
            self._repo.upsert(job)

        return job
