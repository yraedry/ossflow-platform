"""Hub SSE: cola asyncio.Queue por job_id.

Reemplaza el ``_job_events: dict[str, asyncio.Queue]`` global del antiguo
``api/app.py``. Lo usa ``LegacyJobsService`` para alimentar el endpoint
``/api/jobs/{id}/events``. ``BackgroundJobsService`` no lo necesita
(sus consumidores hacen polling).
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

log = logging.getLogger(__name__)

# Intervalo de keepalive del SSE. El frontend tolera ~30s sin datos antes
# de cerrar la conexión; 15s deja margen.
KEEPALIVE_SECONDS = 15.0


class SseHub:
    """Pub/sub in-memory de eventos por job_id."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}

    def register(self, job_id: str) -> asyncio.Queue:
        """Crea (o reutiliza) la cola asociada al job_id."""
        q = self._queues.get(job_id)
        if q is None:
            q = asyncio.Queue()
            self._queues[job_id] = q
        return q

    def get(self, job_id: str) -> Optional[asyncio.Queue]:
        return self._queues.get(job_id)

    def publish(self, job_id: str, event: dict) -> None:
        """Pone ``event`` en la cola si existe.

        ``put_nowait`` para no bloquear al productor (los consumidores
        deben drenar a tiempo; las colas son ilimitadas).
        """
        q = self._queues.get(job_id)
        if q is None:
            return
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover — colas son unbounded
            log.warning("SSE queue full for job %s; event dropped", job_id)

    async def subscribe(
        self,
        job_id: str,
        *,
        keepalive_seconds: float = KEEPALIVE_SECONDS,
    ) -> AsyncIterator[Optional[dict]]:
        """Drena la cola con keepalive periódico.

        Yield ``None`` cada ``keepalive_seconds`` sin eventos para que el
        consumidor pueda emitir un comentario SSE (``: keepalive\\n\\n``).
        Yield el evento (dict) cuando hay uno disponible.
        """
        q = self.register(job_id)
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=keepalive_seconds)
                yield event
            except asyncio.TimeoutError:
                yield None  # señal de keepalive

    def close(self, job_id: str) -> None:
        """Libera la cola asociada al job_id (cuando termina el job)."""
        self._queues.pop(job_id, None)

    def known_ids(self) -> list[str]:
        return list(self._queues.keys())
