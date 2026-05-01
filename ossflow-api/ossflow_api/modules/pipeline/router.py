"""Endpoints HTTP del módulo pipeline.

Migrado de ``api/pipeline.py`` en T_LATE_2.6. Los endpoints leen el
state container (`_pipelines`, `_pipeline_subscribers`, etc.) **lazy**
via ``import api.pipeline as _pmod`` para preservar los monkeypatches
de los tests sobre el shim.

Endpoints:

* ``POST /api/pipeline``                    — crear y lanzar pipeline.
* ``GET  /api/pipeline``                    — listar (summary, paginado).
* ``GET  /api/pipeline/eta``                — ETA estimado por step.
* ``POST /api/pipeline/flush-gpu``          — restart subtitle-generator.
* ``POST /api/pipeline/flush-ollama``       — descargar modelo Ollama.
* ``POST /api/pipeline/pull-ollama-model``  — pull SSE.
* ``POST /api/pipeline/batch``              — lanzar batch multi-season.
* ``GET  /api/pipeline/batch``              — listar batches.
* ``GET  /api/pipeline/batch/{id}``         — detalle batch.
* ``POST /api/pipeline/batch/{id}/cancel``  — cancelar batch.
* ``GET  /api/pipeline/{id}``               — detalle pipeline.
* ``GET  /api/pipeline/{id}/events``        — SSE replay+live.
* ``POST /api/pipeline/{id}/cancel``        — cancelar pipeline.
* ``POST /api/pipeline/{id}/retry``         — re-run con mismas opciones.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .schemas import (
    BatchInfo,
    PipelineInfo,
    StepInfo,
    StepStatus,
    VALID_STEPS,
    serialize_batch,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


def _shim():
    """Resuelve el shim ``api.pipeline`` perezosamente.

    El router accede al state container y a los wrappers via el shim
    para que los tests que parchean ``api.pipeline._pipelines``,
    ``api.pipeline._run_pipeline`` etc. afecten también al router.
    """
    import api.pipeline as _pmod  # noqa: PLC0415
    return _pmod


# ---------------------------------------------------------------------------
# Launch helper (compartido por POST "" y retry)
# ---------------------------------------------------------------------------


def launch_pipeline_internal(
    path: str, steps_raw: list[str], options: dict,
) -> tuple[Optional[PipelineInfo], Optional[dict], int]:
    """Valida inputs y lanza el task del pipeline.

    Devuelve ``(pipeline, error_payload, status_code)``. Éxito:
    ``error_payload`` es None y status 200; fallo: pipeline es None.
    """
    pmod = _shim()
    if not path:
        return None, {"error": "Missing 'path'"}, 400
    if not Path(path).exists():
        return None, {"error": f"Path not accessible: {path}"}, 422
    if not steps_raw:
        return None, {"error": "Missing 'steps' list"}, 400

    invalid = [s for s in steps_raw if s not in VALID_STEPS]
    if invalid:
        return (
            None,
            {"error": f"Invalid steps: {invalid}. Valid: {sorted(VALID_STEPS)}"},
            422,
        )

    GPU_STEPS = {"subtitles", "dubbing"}
    requested_gpu = GPU_STEPS.intersection(steps_raw)
    if requested_gpu:
        for p in pmod._pipelines.values():
            if p.status == StepStatus.RUNNING:
                active_gpu = GPU_STEPS.intersection(s.name for s in p.steps)
                if active_gpu:
                    return (
                        None,
                        {
                            "error": "GPU ocupada",
                            "detail": (
                                f"Pipeline {p.pipeline_id} ya está usando GPU "
                                f"({', '.join(sorted(active_gpu))}). "
                                "Espera a que termine."
                            ),
                            "active_pipeline_id": p.pipeline_id,
                        },
                        409,
                    )

    pipeline_id = str(uuid.uuid4())[:8]
    steps = [StepInfo(name=s) for s in steps_raw]
    pipeline = PipelineInfo(
        pipeline_id=pipeline_id,
        path=path,
        steps=steps,
        options=options,
    )
    pmod._pipelines[pipeline_id] = pipeline
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(pmod._run_pipeline(pipeline, queue))
    pmod._pipeline_tasks[pipeline_id] = task
    pmod._save_history()
    return pipeline, None, 200


# ---------------------------------------------------------------------------
# Maintenance: flush GPU / Ollama / pull-ollama-model
# ---------------------------------------------------------------------------


@router.post("/flush-gpu")
async def flush_gpu():
    """Restart subtitle-generator to free VRAM, then wait until healthy (max 60s)."""
    import httpx
    pmod = _shim()
    subs_url = pmod.subs_client().base_url
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{subs_url}/maintenance/restart")
    except Exception:
        pass
    for _ in range(30):
        await asyncio.sleep(2.0)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{subs_url}/health")
                if r.status_code == 200:
                    return {"ok": True, "message": "subtitle-generator restarted and healthy"}
        except Exception:
            pass
    return JSONResponse(
        {"ok": False, "message": "subtitle-generator did not recover in 60s"},
        status_code=503,
    )


@router.post("/flush-ollama")
async def flush_ollama() -> dict:
    """Descarga el modelo de Ollama de VRAM al instante."""
    import httpx
    from api.settings import get_setting

    model = get_setting("translation_model") or "qwen2.5:7b-instruct-q4_K_M"
    base = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{base}/api/chat",
                json={"model": model, "messages": [], "keep_alive": 0, "stream": False},
            )
        return {"ok": r.status_code < 400, "status": r.status_code}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/pull-ollama-model")
async def pull_ollama_model():
    """Descarga el modelo Ollama configurado. Devuelve SSE con progreso."""
    import httpx
    from api.settings import get_setting

    model = get_setting("translation_model") or "qwen2.5:7b-instruct-q4_K_M"
    base = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")

    async def _stream():
        try:
            async with httpx.AsyncClient(timeout=1800.0) as client:
                async with client.stream(
                    "POST",
                    f"{base}/api/pull",
                    json={"name": model, "stream": True},
                ) as r:
                    if r.status_code >= 400:
                        body = await r.aread()
                        yield f"data: {json.dumps({'status': 'error', 'error': body.decode()[:200]})}\n\n"
                        return
                    async for line in r.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue
                        if data.get("error"):
                            yield f"data: {json.dumps({'status': 'error', 'error': data['error']})}\n\n"
                            return
                        total = data.get("total")
                        completed = data.get("completed")
                        event = {"status": data.get("status", "")}
                        if total:
                            event["total"] = total
                        if completed:
                            event["completed"] = completed
                            event["pct"] = round(completed / total * 100, 1) if total else 0
                        yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'status': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# ETA
# ---------------------------------------------------------------------------


@router.get("/eta")
async def pipeline_eta(
    steps: str = "",
    video_duration_sec: Optional[float] = None,
    path: Optional[str] = None,
):
    """Estima ETA por step a partir del histórico de pipelines completados."""
    pmod = _shim()
    requested = [s.strip() for s in steps.split(",") if s.strip()]
    if not requested:
        requested = sorted(VALID_STEPS)
    invalid = [s for s in requested if s not in VALID_STEPS]
    if invalid:
        return JSONResponse(
            {"error": f"Invalid steps: {invalid}. Valid: {sorted(VALID_STEPS)}"},
            status_code=422,
        )

    if video_duration_sec is None and path:
        video_duration_sec = pmod._total_video_duration(path)

    MIN_SAMPLES = 3
    WINDOW = 20

    by_step: dict[str, list[float]] = {s: [] for s in requested}
    ordered = sorted(pmod._pipelines.values(), key=lambda p: p.created_at, reverse=True)
    for pipe in ordered:
        for s in pipe.steps:
            if s.name not in by_step:
                continue
            if s.status != StepStatus.COMPLETED:
                continue
            dur = pmod._duration_seconds(s.started_at, s.completed_at)
            if dur is None:
                continue
            by_step[s.name].append(dur)

    per_step: dict[str, Optional[float]] = {}
    total = 0.0
    total_known = True
    for name in requested:
        samples = by_step[name][:WINDOW]
        if len(samples) < MIN_SAMPLES:
            per_step[name] = None
            total_known = False
            continue
        est = pmod._median(samples)
        per_step[name] = est
        total += est

    return {
        "per_step": per_step,
        "total_seconds": total if total_known else None,
        "video_duration_sec": video_duration_sec,
        "sample_counts": {k: len(by_step[k]) for k in requested},
    }


# ---------------------------------------------------------------------------
# Pipeline CRUD + control
# ---------------------------------------------------------------------------


@router.post("")
async def create_pipeline(request: Request):
    """Crea y lanza un pipeline.

    Body::
        {
            "path": "/data/.../video.mkv",
            "steps": ["chapters", "subtitles", "translate", "dubbing"],
            "options": { "dry_run": false, "voice_profile": "gordon_ryan" }
        }
    """
    pmod = _shim()
    body = await request.json()
    path = body.get("path", "")
    steps_raw = body.get("steps", [])
    options = body.get("options", {})

    pipeline, err, code = pmod._launch_pipeline_internal(path, steps_raw, options)
    if err is not None:
        return JSONResponse(err, status_code=code)

    return {
        "pipeline_id": pipeline.pipeline_id,
        "steps": [s.name for s in pipeline.steps],
        "status": pipeline.status.value,
    }


# ---------------------------------------------------------------------------
# Batch (multi-season). Registrado ANTES de /{pipeline_id} para que el path
# literal /batch matchee antes que el catch-all.
# ---------------------------------------------------------------------------


@router.post("/batch")
async def create_batch(request: Request):
    """Lanza un batch multi-season server-side."""
    pmod = _shim()
    body = await request.json()
    name = (body.get("name") or "").strip() or "batch"
    paths = body.get("paths") or []
    steps = body.get("steps") or []
    options = body.get("options") or {}
    continue_on_fail = bool(body.get("continue_on_fail", True))

    if not paths or not isinstance(paths, list):
        return JSONResponse({"error": "Missing or invalid 'paths' list"}, status_code=400)
    if not steps:
        return JSONResponse({"error": "Missing 'steps' list"}, status_code=400)

    invalid_steps = [s for s in steps if s not in VALID_STEPS]
    if invalid_steps:
        return JSONResponse(
            {"error": f"Invalid steps: {invalid_steps}. Valid: {sorted(VALID_STEPS)}"},
            status_code=422,
        )

    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        return JSONResponse(
            {"error": "Some paths do not exist", "missing": missing},
            status_code=422,
        )

    batch_id = str(uuid.uuid4())[:8]
    batch = BatchInfo(
        batch_id=batch_id,
        name=name,
        paths=list(paths),
        steps=list(steps),
        options=dict(options),
        continue_on_fail=continue_on_fail,
    )
    pmod._batches[batch_id] = batch
    pmod._batch_cancel[batch_id] = False
    pmod._batch_tasks[batch_id] = asyncio.create_task(pmod._run_batch(batch))

    return serialize_batch(batch)


@router.get("/batch")
async def list_batches(limit: int = 20):
    pmod = _shim()
    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    ordered = sorted(pmod._batches.values(), key=lambda b: b.created_at, reverse=True)[:limit]
    return {"batches": [serialize_batch(b) for b in ordered]}


@router.get("/batch/{batch_id}")
async def get_batch(batch_id: str):
    pmod = _shim()
    batch = pmod._batches.get(batch_id)
    if not batch:
        return JSONResponse({"error": "Batch not found"}, status_code=404)
    return serialize_batch(batch)


@router.post("/batch/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    pmod = _shim()
    batch = pmod._batches.get(batch_id)
    if not batch:
        return JSONResponse({"error": "Batch not found"}, status_code=404)
    if batch.status not in (StepStatus.RUNNING, StepStatus.PENDING):
        return JSONResponse(
            {"error": f"Batch is {batch.status.value}, cannot cancel"},
            status_code=409,
        )
    pmod._batch_cancel[batch_id] = True
    if batch.pipeline_ids:
        active_pid = batch.pipeline_ids[-1]
        active = pmod._pipelines.get(active_pid)
        if active and active.status in (StepStatus.RUNNING, StepStatus.PENDING):
            pmod._pipeline_cancel[active_pid] = True
            t = pmod._pipeline_tasks.get(active_pid)
            if t and not t.done():
                t.cancel()
    return {"batch_id": batch_id, "status": "cancelling"}


@router.get("/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Devuelve el estado actual de un pipeline."""
    pmod = _shim()
    pipeline = pmod._pipelines.get(pipeline_id)
    if not pipeline:
        return JSONResponse({"error": "Pipeline not found"}, status_code=404)
    return {
        "pipeline_id": pipeline.pipeline_id,
        "path": pipeline.path,
        "status": pipeline.status.value,
        "current_step": pipeline.current_step,
        "steps": [
            {
                "name": s.name,
                "status": s.status.value,
                "progress": s.progress,
                "message": s.message,
                "started_at": s.started_at,
                "completed_at": s.completed_at,
                "diff": s.diff,
            }
            for s in pipeline.steps
        ],
        "options": pipeline.options,
        "created_at": pipeline.created_at,
        "completed_at": pipeline.completed_at,
    }


@router.get("/{pipeline_id}/events")
async def pipeline_events(pipeline_id: str):
    """SSE replay + live para el progreso de un pipeline."""
    pmod = _shim()
    pipeline = pmod._pipelines.get(pipeline_id)
    if not pipeline:
        return JSONResponse({"error": "Pipeline not found"}, status_code=404)

    queue = pmod._subscribe(pipeline_id)
    replay = list(pipeline.log_buffer)
    replay_max_seq = max((e.get("seq", 0) for e in replay), default=0)
    terminal = pipeline.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.CANCELLED)

    async def event_stream():
        try:
            for evt in replay:
                yield f"data: {json.dumps(evt)}\n\n"
            if terminal:
                return
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if data.get("seq", 0) and data["seq"] <= replay_max_seq:
                    continue
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("type") in ("pipeline_completed", "pipeline_failed"):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            pmod._unsubscribe(pipeline_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{pipeline_id}/cancel")
async def cancel_pipeline(pipeline_id: str):
    """Pide cancelación de un pipeline activo."""
    pmod = _shim()
    pipeline = pmod._pipelines.get(pipeline_id)
    if not pipeline:
        return JSONResponse({"error": "Pipeline not found"}, status_code=404)
    if pipeline.status not in (StepStatus.RUNNING, StepStatus.PENDING):
        return JSONResponse(
            {"error": f"Pipeline is {pipeline.status.value}, cannot cancel"},
            status_code=409,
        )
    pmod._pipeline_cancel[pipeline_id] = True
    task = pmod._pipeline_tasks.get(pipeline_id)
    if task and not task.done():
        task.cancel()
    return {"pipeline_id": pipeline_id, "status": "cancelling"}


@router.post("/{pipeline_id}/retry")
async def retry_pipeline(pipeline_id: str):
    """Re-ejecuta un pipeline finalizado conservando path/steps/options."""
    pmod = _shim()
    src = pmod._pipelines.get(pipeline_id)
    if not src:
        return JSONResponse({"error": "Pipeline not found"}, status_code=404)
    if src.status in (StepStatus.RUNNING, StepStatus.PENDING):
        return JSONResponse(
            {"error": "Pipeline still active, cannot retry"},
            status_code=409,
        )
    new_id = str(uuid.uuid4())[:8]
    new_steps = [StepInfo(name=s.name) for s in src.steps]
    new_pipe = PipelineInfo(
        pipeline_id=new_id,
        path=src.path,
        steps=new_steps,
        options=dict(src.options),
    )
    pmod._pipelines[new_id] = new_pipe
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(pmod._run_pipeline(new_pipe, queue))
    pmod._pipeline_tasks[new_id] = task
    return {
        "pipeline_id": new_id,
        "retried_from": pipeline_id,
        "steps": [s.name for s in new_steps],
        "status": new_pipe.status.value,
    }


@router.get("")
async def list_pipelines(
    response: Response,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
):
    """Lista pipelines (summary, sorted by created_at desc).

    Query params:
        limit, offset — paginación.
        status — filtro opcional (pending/running/completed/failed/cancelled).

    Response incluye header ``X-Total-Count`` con el total filtrado.
    Para el payload completo (step diff, message) usar
    ``GET /api/pipeline/{id}``.
    """
    pmod = _shim()
    if limit < 1:
        limit = 1
    if limit > 500:
        limit = 500
    if offset < 0:
        offset = 0

    ordered = sorted(pmod._pipelines.values(), key=lambda p: p.created_at, reverse=True)
    if status:
        ordered = [p for p in ordered if p.status.value == status]
    total = len(ordered)
    page = ordered[offset : offset + limit]

    response.headers["X-Total-Count"] = str(total)
    return {
        "pipelines": [
            {
                "pipeline_id": p.pipeline_id,
                "path": p.path,
                "status": p.status.value,
                "created_at": p.created_at,
                "completed_at": p.completed_at,
                "steps": [
                    {
                        "name": s.name,
                        "status": s.status.value,
                        "progress": s.progress,
                    }
                    for s in p.steps
                ],
            }
            for p in page
        ]
    }
