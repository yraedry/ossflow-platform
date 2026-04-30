"""Utilidad común: recuperación de jobs huérfanos al arrancar.

Cuando el servidor cae con jobs en ``RUNNING`` o ``QUEUED``, esos jobs no
van a volver a ejecutarse — quedan congelados en BD. Al arrancar
marcamos ``FAILED`` con un mensaje explicativo para que el usuario sepa
qué pasó (en lugar de ver una barra de progreso muerta para siempre).

Ambos repositorios (background y legacy) llaman a esta función al
``init()``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Protocol

from ..models import JobStatus

ORPHAN_ERROR_MESSAGE = "interrupted: server restarted"


class _HasStatus(Protocol):
    status: str


def mark_running_as_failed(
    jobs: Iterable[_HasStatus],
    setter: Callable[[_HasStatus, str, str, str], None],
) -> int:
    """Marca jobs en ``RUNNING``/``QUEUED`` como ``FAILED``.

    ``setter(job, status, error, completed_at)`` es la función que el
    repositorio expone para mutar+persistir un job individual. Hacer
    inyección en lugar de tocar SQL aquí mantiene esta función pura y
    testeable sin BD.

    Devuelve el número de jobs modificados.
    """
    interrupted_states = {JobStatus.RUNNING.value, JobStatus.QUEUED.value}
    now = datetime.now().isoformat()
    count = 0
    for job in jobs:
        if job.status in interrupted_states:
            setter(job, JobStatus.FAILED.value, ORPHAN_ERROR_MESSAGE, now)
            count += 1
    return count
