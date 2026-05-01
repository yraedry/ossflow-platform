"""Router HTTP de ``/api/jobs/*`` (legacy con SSE).

Mantiene el contrato exacto del antiguo ``api/app.py``:

* ``POST /api/jobs`` — crea un job, lanza el runner correspondiente,
  devuelve ``{"job_id": "...", "status": "queued"}``.
* ``GET /api/jobs`` — lista todos.
* ``GET /api/jobs/{id}`` — uno o 404.
* ``GET /api/jobs/{id}/events`` — SSE stream con el formato byte-exact
  ``data: {...}\\n\\n`` y ``: keepalive\\n\\n``.

El cierre del acoplamiento #7 (elevenlabs ↔ jobs) llega en T19.6: el
servicio legacy se inyecta en ``ElevenLabsService`` por DI.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from ..dependencies import (
    JobRunner,
    get_legacy_jobs_dispatch_table,
    get_legacy_jobs_service,
)
from ..schemas import (
    LegacyJobCreateRequest,
    LegacyJobCreateResponse,
    LegacyJobListResponse,
    LegacyJobResponse,
)
from ..services.legacy import LegacyJobsService

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("", response_model=LegacyJobCreateResponse)
async def create_job(
    request: Request,
    body: LegacyJobCreateRequest,
    svc: LegacyJobsService = Depends(get_legacy_jobs_service),
    dispatch: dict[str, JobRunner] = Depends(get_legacy_jobs_dispatch_table),
) -> dict:
    """Crea un job legacy y lanza el runner correspondiente."""
    runner = dispatch.get(body.type)
    if runner is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown job type '{body.type}'. Allowed: {sorted(dispatch.keys())}",
        )

    job = svc.register_job(body.type, body.path)

    # Body completo (incluye campos extra como voice_profile/use_model_voice).
    raw = await request.json()
    raw = raw if isinstance(raw, dict) else {}

    async def _run(job_arg):
        await runner(job_arg, raw)

    svc.spawn_runner(job, _run)
    return {"job_id": job.job_id, "status": job.status}


@router.get("", response_model=LegacyJobListResponse)
async def list_jobs(
    type: str | None = None,
    svc: LegacyJobsService = Depends(get_legacy_jobs_service),
) -> dict:
    return {"jobs": [j.to_dict() for j in svc.list_all(type_filter=type)]}


@router.get("/{job_id}", response_model=LegacyJobResponse)
async def get_job(
    job_id: str,
    svc: LegacyJobsService = Depends(get_legacy_jobs_service),
) -> dict:
    job = svc.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.get("/{job_id}/events")
async def stream_events(
    job_id: str,
    svc: LegacyJobsService = Depends(get_legacy_jobs_service),
) -> StreamingResponse:
    """Stream SSE de los eventos del job. Mantiene el formato legacy:

    * ``data: {...}\\n\\n`` para cada evento.
    * ``: keepalive\\n\\n`` cada ``KEEPALIVE_SECONDS`` sin actividad.
    """
    job = svc.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def _generator() -> AsyncIterator[bytes]:
        async for evt in svc.subscribe_events(job_id):
            if evt is None:
                yield b": keepalive\n\n"
                continue
            payload = json.dumps(evt, ensure_ascii=False, default=str)
            yield f"data: {payload}\n\n".encode("utf-8")
            # Si el evento es terminal, cerramos la cola y rompemos.
            status = evt.get("status") if isinstance(evt, dict) else None
            if status in ("completed", "failed"):
                svc.close_events(job_id)
                break

    return StreamingResponse(_generator(), media_type="text/event-stream")
