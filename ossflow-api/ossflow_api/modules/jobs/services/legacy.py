"""Servicio público de "legacy jobs".

Sustituye al estado global de ``api/app.py`` (``_jobs``, ``_job_events``,
``_persist_job``, ``_emit``) y a la persistencia ad-hoc de
``api/jobs_store.JobsStore``.

Composición:
* ``LegacyJobsRepository`` → persistencia SQL.
* ``JobsScheduler.run_task`` → lanza el coroutine en el loop actual
  (no en un thread propio: SSE necesita compartir loop con el endpoint).
* ``SseHub`` → cola asyncio.Queue por job_id.

API pública pensada para sustituir el patrón antiguo:

* ``register_job(job_type, video_path)`` — crea el job y registra cola SSE.
* ``spawn_runner(job, runner)`` — lanza ``runner(job)`` como ``asyncio.Task``.
* ``get(job_id)`` / ``list_all(type_filter)``.
* ``update_progress(job_id, progress, message)`` — sin persistir (rápido).
* ``update_status(job_id, status, *, result=None)`` — persiste + emite.
* ``emit(job_id, event)`` — empuja al hub SSE; persiste si el evento
  contiene ``status``.
* ``subscribe_events(job_id)`` — async iterator para el endpoint SSE.
* ``init()`` — hook de startup (recovery de huérfanos).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from ..models import JobStatus, LegacyJob
from .._internal.scheduler import JobsScheduler
from .._internal.sse_hub import SseHub
from ..repositories.legacy import LegacyJobsRepository

log = logging.getLogger(__name__)

JobRunner = Callable[[LegacyJob], Awaitable[None]]


class LegacyJobsService:
    """Encola y gestiona jobs estilo "legacy" con persistencia + SSE."""

    def __init__(
        self,
        repo: LegacyJobsRepository,
        scheduler: JobsScheduler,
        sse_hub: SseHub,
    ) -> None:
        self._repo = repo
        self._scheduler = scheduler
        self._sse_hub = sse_hub
        self._initialized = False

    # --- arranque ---------------------------------------------------------

    def init(self) -> None:
        if self._initialized:
            return
        recovered = self._repo.init_db_and_recover()
        if recovered:
            log.info("Legacy jobs: %d huérfanos marcados como FAILED", recovered)
        self._initialized = True

    # --- creación + scheduling -------------------------------------------

    def register_job(self, job_type: str, video_path: str) -> LegacyJob:
        """Crea el job en estado QUEUED, lo persiste y registra cola SSE."""
        job_id = str(uuid.uuid4())[:8]
        job = LegacyJob(
            job_id=job_id,
            job_type=job_type,
            video_path=video_path,
            status=JobStatus.QUEUED.value,
        )
        self._repo.upsert(job)
        self._sse_hub.register(job_id)
        return job

    def spawn_runner(self, job: LegacyJob, runner: JobRunner) -> None:
        """Lanza ``runner(job)`` como ``asyncio.Task`` en el loop actual.

        El runner debe usar ``self.update_progress``, ``self.update_status``
        y ``self.emit`` para reportar avances. Excepciones no controladas
        marcan el job como FAILED.
        """

        async def _wrapped() -> None:
            try:
                await runner(job)
            except Exception as exc:  # noqa: BLE001
                log.exception("Legacy job %s failed", job.job_id)
                self.update_status(
                    job.job_id,
                    JobStatus.FAILED,
                    message=f"{exc.__class__.__name__}: {exc}",
                )
                await self.emit(job.job_id, {
                    "status": JobStatus.FAILED.value,
                    "message": f"{exc.__class__.__name__}: {exc}",
                })

        self._scheduler.run_task(_wrapped(), name=f"legjob-{job.job_id}")

    # --- consultas -------------------------------------------------------

    def get(self, job_id: str) -> Optional[LegacyJob]:
        return self._repo.get(job_id)

    def list_all(self, type_filter: Optional[str] = None) -> list[LegacyJob]:
        return self._repo.list_all(type_filter=type_filter)

    # --- mutaciones ------------------------------------------------------

    def update_progress(
        self,
        job_id: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        """Actualiza progreso y/o mensaje. NO persiste (sería costoso en
        loops de progreso). El próximo ``update_status`` o ``emit`` con
        status sí persistirá."""
        job = self._repo.get(job_id)
        if job is None:
            return
        if progress is not None:
            try:
                job.progress = float(progress)
            except (TypeError, ValueError):
                pass
        if message:
            job.message = message
        # No upsert aquí — sería write-amplification. Solo se persiste en
        # cambios de status. Sí guardamos en memoria a través de get/upsert
        # vía emit() cuando el evento incluya un status terminal.

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        message: Optional[str] = None,
        result: Optional[dict] = None,
        progress: Optional[float] = None,
    ) -> None:
        """Persiste un cambio de status (incluye result en COMPLETED)."""
        job = self._repo.get(job_id)
        if job is None:
            return
        job.status = status.value
        if message is not None:
            job.message = message
        if result is not None:
            job.result = result
        if progress is not None:
            job.progress = progress
        if status in (JobStatus.COMPLETED, JobStatus.FAILED):
            if not job.completed_at:
                job.completed_at = datetime.now().isoformat()
            if status == JobStatus.COMPLETED:
                if job.progress is None or job.progress < 100.0:
                    job.progress = 100.0
        self._repo.upsert(job)

    async def emit(self, job_id: str, event: dict[str, Any]) -> None:
        """Empuja ``event`` a la cola SSE; si tiene ``status``, persiste.

        Replica el comportamiento del antiguo ``_emit`` de ``app.py``.
        """
        # Si el evento trae status, persistimos antes de publicar para
        # que un suscriptor que vea el evento pueda hacer GET y obtener
        # el estado actualizado.
        if "status" in event:
            try:
                new_status = JobStatus(event["status"])
            except ValueError:
                new_status = None
            if new_status is not None:
                self.update_status(
                    job_id,
                    new_status,
                    message=event.get("message"),
                    result=event.get("result"),
                    progress=event.get("progress"),
                )
        elif "progress" in event or "message" in event:
            # Solo progreso: actualiza en memoria sin persistir.
            self.update_progress(
                job_id,
                progress=event.get("progress"),
                message=event.get("message"),
            )
        self._sse_hub.publish(job_id, event)

    # --- SSE -------------------------------------------------------------

    async def subscribe_events(self, job_id: str) -> AsyncIterator[Optional[dict]]:
        """Drena los eventos del job. Yield ``None`` en cada keepalive."""
        async for evt in self._sse_hub.subscribe(job_id):
            yield evt

    def close_events(self, job_id: str) -> None:
        """Libera la cola SSE. Llamar cuando el job termina."""
        self._sse_hub.close(job_id)
