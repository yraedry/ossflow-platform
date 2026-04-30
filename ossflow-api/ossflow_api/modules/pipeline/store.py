"""SSE fan-out + buffer del pipeline.

Migrado de ``api/pipeline.py`` en T_LATE_2.5b. Funciones puras que
reciben los dicts de estado por parámetro — el state container
(``_pipelines``, ``_pipeline_subscribers``, ``_batches``, etc.) se
mantiene como globals del shim ``api/pipeline.py`` porque los tests
parchean esos atributos directamente en el módulo del shim.

* ``subscribe(subscribers, pipeline_id)`` registra una nueva queue.
* ``unsubscribe(subscribers, pipeline_id, q)`` la descadena.
* ``emit(pipeline, subscribers, event)`` añade al buffer y broadcast
  a todos los subscribers vivos.

El argumento ``queue`` legacy de ``_emit`` se mantiene en el shim por
retrocompat con call sites pero no se usa — fan-out a TODOS los
subscribers evita que un consumer "robe" eventos a otro.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .schemas import PipelineInfo


LOG_BUFFER_CAP = 2000


def subscribe(
    subscribers: dict[str, list[asyncio.Queue]],
    pipeline_id: str,
) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    subscribers.setdefault(pipeline_id, []).append(q)
    return q


def unsubscribe(
    subscribers: dict[str, list[asyncio.Queue]],
    pipeline_id: str,
    q: asyncio.Queue,
) -> None:
    subs = subscribers.get(pipeline_id)
    if not subs:
        return
    try:
        subs.remove(q)
    except ValueError:
        pass
    if not subs:
        subscribers.pop(pipeline_id, None)


async def emit(
    pipeline: PipelineInfo,
    subscribers: dict[str, list[asyncio.Queue]],
    event: dict[str, Any],
) -> None:
    """Añade al buffer persistente (capped) y broadcast a todos los
    subscribers vivos del pipeline.

    Cada evento lleva un ``seq`` monotónico para dedupe client-side
    tras reconnect (el replay del buffer entregaría eventos repetidos
    sin esa marca).
    """
    pipeline.event_seq += 1
    tagged = {**event, "seq": pipeline.event_seq}
    pipeline.log_buffer.append(tagged)
    if len(pipeline.log_buffer) > LOG_BUFFER_CAP:
        del pipeline.log_buffer[: len(pipeline.log_buffer) - LOG_BUFFER_CAP]
    for q in list(subscribers.get(pipeline.pipeline_id, [])):
        try:
            q.put_nowait(tagged)
        except asyncio.QueueFull:  # pragma: no cover - unbounded queues
            pass
