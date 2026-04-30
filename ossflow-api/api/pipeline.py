"""Pipeline endpoint: execute multiple processing steps sequentially.

Each step delegates to a backend microservice over HTTP (see
``api.backend_client``). If any step fails, the pipeline stops and reports
the error. Progress is streamed via SSE.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from api.event_normalizer import normalize
from api.paths import to_container_path
from api.settings import get_library_path
from api.backend_client import (
    BackendClient,
    BackendError,
    dubbing_client,
    splitter_client,
    subs_client,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

BASE_DIR = Path(__file__).parent.parent  # processor-api root

from api.settings import CONFIG_DIR as _CONFIG_DIR
HISTORY_FILE = _CONFIG_DIR / "pipeline_history.json"


# ---------------------------------------------------------------------------
# Data models — migrados a modules/pipeline/schemas.py (T_LATE_2.1).
# Los tests parchean ``api.pipeline.PipelineInfo`` y similares, así que se
# re-exportan desde aquí.
# ---------------------------------------------------------------------------

from ossflow_api.modules.pipeline.schemas import (  # noqa: F401,E402
    BatchInfo,
    PipelineInfo,
    StepInfo,
    StepStatus,
    serialize_batch as _serialize_batch_new,
    serialize_pipeline as _serialize_new,
)


# In-memory store
_pipelines: dict[str, PipelineInfo] = {}
_batches: dict[str, BatchInfo] = {}
_batch_tasks: dict[str, asyncio.Task] = {}
_batch_cancel: dict[str, bool] = {}
# Per-pipeline list of subscriber queues. Fan-out: each SSE client gets its
# own queue so multiple consumers (e.g. StrictMode double-mount, reconnect
# while a previous EventSource is still draining) do NOT steal events from
# each other. A single shared queue caused "missing live logs" in LogPanel.
_pipeline_subscribers: dict[str, list[asyncio.Queue]] = {}
_pipeline_tasks: dict[str, asyncio.Task] = {}
_pipeline_cancel: dict[str, bool] = {}


# SSE primitives — migradas a modules/pipeline/store.py (T_LATE_2.5b).
# El state (_pipeline_subscribers) sigue siendo global del shim porque los
# tests lo parchean por nombre.
from ossflow_api.modules.pipeline import store as _store_mod  # noqa: E402


def _subscribe(pipeline_id: str) -> asyncio.Queue:
    return _store_mod.subscribe(_pipeline_subscribers, pipeline_id)


def _unsubscribe(pipeline_id: str, q: asyncio.Queue) -> None:
    _store_mod.unsubscribe(_pipeline_subscribers, pipeline_id, q)


def _serialize(p: PipelineInfo) -> dict:
    return _serialize_new(p)


# Debounced history save — migrado a modules/pipeline/history.py (T_LATE_2.2).
# Mantenemos wrappers wrapper a nivel módulo para que los tests sigan
# parcheando ``pmod._save_history`` y ``pmod.HISTORY_FILE`` directamente.
from ossflow_api.modules.pipeline import history as _history_mod  # noqa: E402

_SAVE_MIN_INTERVAL = _history_mod.SAVE_MIN_INTERVAL


def _write_history_sync() -> None:
    _history_mod.write_history_sync(_pipelines, HISTORY_FILE)


def _save_history() -> None:
    _history_mod.save_history(_pipelines, HISTORY_FILE)


def _load_history() -> None:
    _history_mod.load_history(_pipelines, HISTORY_FILE)


_load_history()

# Valid step names — re-exportadas desde modules/pipeline/schemas (T_LATE_2.1)
from ossflow_api.modules.pipeline.schemas import (  # noqa: F401,E402
    STEP_ORDER,
    VALID_STEPS,
)


# ---------------------------------------------------------------------------
# Step execution helpers
# ---------------------------------------------------------------------------

SIDECAR_NAME = ".bjj-meta.json"


# Backend dispatch — migrado a modules/pipeline/backend_dispatch.py
# (T_LATE_2.4). Re-export con prefijo _ para preservar la API que parchean
# los tests.
from ossflow_api.modules.pipeline.backend_dispatch import (  # noqa: E402,F401
    client_and_payload as _client_and_payload_new,
    load_oracle_for_path as _load_oracle_for_path,
    load_voice_profile_for_path as _load_voice_profile_for_path,
)


def _client_and_payload(
    step_name: str,
    path: str,
    options: dict,
    chained_path: Optional[str] = None,
) -> tuple[BackendClient, dict, bool]:
    """Wrapper de retrocompat — la lógica vive en
    ``modules/pipeline/backend_dispatch.py``. Tests parchean
    ``api.pipeline._client_and_payload`` directamente.
    """
    return _client_and_payload_new(step_name, path, options, chained_path)


# Skip-detection y diff — migrados a modules/pipeline/{skip_detector,diff}.py
# (T_LATE_2.3). Re-export con prefijo _ para preservar la API que parchean
# los tests (api.pipeline._chapter_has_*, _detect_season_folder, etc.).
from ossflow_api.modules.pipeline.skip_detector import (  # noqa: E402,F401
    CHAPTER_SE_RE as _CHAPTER_SE_RE,
    DUB_SUFFIXES as _DUB_SUFFIXES,
    chapter_has_en_subs as _chapter_has_en_subs,
    chapter_has_es_subs as _chapter_has_es_subs,
    chapter_is_dubbed as _chapter_is_dubbed,
    list_chapters as _list_chapters,
    season_already_dubbed as _season_already_dubbed,
    season_already_subbed_en as _season_already_subbed_en,
    season_already_subbed_es as _season_already_subbed_es,
)
from ossflow_api.modules.pipeline.diff import (  # noqa: E402,F401
    SEASON_DIR_RE as _SEASON_DIR_RE,
    VIDEO_EXTS as _VIDEO_EXTS,
    compute_diff as _compute_diff,
    detect_season_folder as _detect_season_folder,
    snapshot_dir as _snapshot_dir,
    target_dir as _target_dir,
)


async def _emit(pipeline: PipelineInfo, queue: asyncio.Queue, event: dict) -> None:
    """Wrapper retrocompat — la lógica vive en
    ``modules/pipeline/store.emit``. ``queue`` se ignora (legacy).
    """
    await _store_mod.emit(pipeline, _pipeline_subscribers, event)


# Runner — migrado a modules/pipeline/runner.py (T_LATE_2.5c).
# Los wrappers _run_step / _run_pipeline / _flush_gpu_after_step /
# _refresh_scan_cache_for delegan al módulo nuevo. Tests parchean estos
# símbolos en el shim, así que se mantienen como funciones nivel-módulo.
from ossflow_api.modules.pipeline import runner as _runner_mod  # noqa: E402


async def _run_step(
    pipeline: PipelineInfo,
    step_index: int,
    queue: asyncio.Queue,
) -> bool:
    return await _runner_mod.run_step(pipeline, step_index, queue)


async def _flush_gpu_after_step(
    pipeline: PipelineInfo, queue: asyncio.Queue,
) -> None:
    await _runner_mod.flush_gpu_after_step(pipeline, queue)


async def _run_pipeline(pipeline: PipelineInfo, queue: asyncio.Queue) -> None:
    await _runner_mod.run_pipeline(pipeline, queue)


def _refresh_scan_cache_for(pipeline_path: str) -> None:
    _runner_mod.refresh_scan_cache_for(pipeline_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# ETA helpers — migrados a modules/pipeline/eta.py (T_LATE_2.5a).
from ossflow_api.modules.pipeline.eta import (  # noqa: E402,F401
    duration_seconds as _duration_seconds,
    median as _median,
    total_video_duration as _total_video_duration,
)


@router.post("/flush-gpu")
async def flush_gpu():
    """Restart subtitle-generator to free VRAM, then wait until healthy (max 60s)."""
    import httpx

    subs_url = subs_client().base_url

    # Fire restart — service dies before responding, so ignore errors
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{subs_url}/maintenance/restart")
    except Exception:
        pass  # expected: service kills itself mid-response

    # Poll /health until up (max 60s)
    for _ in range(30):
        await asyncio.sleep(2.0)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{subs_url}/health")
                if r.status_code == 200:
                    return {"ok": True, "message": "subtitle-generator restarted and healthy"}
        except Exception:
            pass

    return JSONResponse({"ok": False, "message": "subtitle-generator did not recover in 60s"}, status_code=503)


@router.post("/flush-ollama")
async def flush_ollama() -> dict:
    """Descarga el modelo de Ollama de VRAM al instante.

    Útil entre fases del pipeline secuencial: tras translate, antes de Kokoro,
    para liberar VRAM (~4.5 GB con qwen2.5:7b-Q4) y evitar OOM.
    """
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
    """Descarga el modelo Ollama configurado. Devuelve SSE con progreso en tiempo real.

    Cada evento es JSON: {status, total?, completed?, pct?}
    Evento final: {status:"success"} o {status:"error", error:"..."}
    """
    import httpx
    from fastapi.responses import StreamingResponse

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


@router.get("/eta")
async def pipeline_eta(
    steps: str = "",
    video_duration_sec: Optional[float] = None,
    path: Optional[str] = None,
):
    """Estimate per-step and total ETA from historical completed pipelines."""
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
        video_duration_sec = _total_video_duration(path)

    MIN_SAMPLES = 3
    WINDOW = 20

    by_step: dict[str, list[float]] = {s: [] for s in requested}
    ordered = sorted(_pipelines.values(), key=lambda p: p.created_at, reverse=True)
    for pipe in ordered:
        for s in pipe.steps:
            if s.name not in by_step:
                continue
            if s.status != StepStatus.COMPLETED:
                continue
            dur = _duration_seconds(s.started_at, s.completed_at)
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
        est = _median(samples)
        per_step[name] = est
        total += est

    return {
        "per_step": per_step,
        "total_seconds": total if total_known else None,
        "video_duration_sec": video_duration_sec,
        "sample_counts": {k: len(by_step[k]) for k in requested},
    }


def _launch_pipeline_internal(
    path: str, steps_raw: list[str], options: dict
) -> tuple[Optional[PipelineInfo], Optional[dict], int]:
    """Validate inputs and start a pipeline task.

    Returns (pipeline, error_payload, status_code). On success error_payload is
    None and status_code is 200; on failure pipeline is None.
    """
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
        for p in _pipelines.values():
            if p.status == StepStatus.RUNNING:
                active_gpu = GPU_STEPS.intersection(s.name for s in p.steps)
                if active_gpu:
                    return (
                        None,
                        {
                            "error": "GPU ocupada",
                            "detail": f"Pipeline {p.pipeline_id} ya está usando GPU ({', '.join(sorted(active_gpu))}). Espera a que termine.",
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
    _pipelines[pipeline_id] = pipeline
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(_run_pipeline(pipeline, queue))
    _pipeline_tasks[pipeline_id] = task
    _save_history()
    return pipeline, None, 200


@router.post("")
async def create_pipeline(request: Request):
    """Create and launch a processing pipeline.

    Body::

        {
            "path": "/data/instructionals/Some Instructional/video.mkv",
            "steps": ["chapters", "subtitles", "translate", "dubbing"],
            "options": { "dry_run": false, "voice_profile": "gordon_ryan" }
        }
    """
    body = await request.json()
    path = body.get("path", "")
    steps_raw = body.get("steps", [])
    options = body.get("options", {})

    pipeline, err, code = _launch_pipeline_internal(path, steps_raw, options)
    if err is not None:
        return JSONResponse(err, status_code=code)

    return {
        "pipeline_id": pipeline.pipeline_id,
        "steps": [s.name for s in pipeline.steps],
        "status": pipeline.status.value,
    }


# ---------------------------------------------------------------------------
# Batch (multi-season) — server-side orchestration
# ---------------------------------------------------------------------------
#
# Why this exists: the previous "Procesar todo" loop ran in the browser. If the
# user closed the tab between seasons, only the first season completed and the
# rest never started. The server-side batch survives navigator close, machine
# sleep, etc. — only a container restart kills it (handled gracefully: on
# reload the in-flight pipeline is marked FAILED and the batch returns 404).
#
# Routes are registered *above* `/{pipeline_id}` so FastAPI matches the
# literal `/batch` prefix before falling through to the catch-all path param.


def _serialize_batch(b: BatchInfo) -> dict:
    return _serialize_batch_new(b)


async def _run_batch(batch: BatchInfo) -> None:
    """Sequentially launch + await one pipeline per season path."""
    batch.status = StepStatus.RUNNING
    log.info("[batch %s] starting %d seasons", batch.batch_id, len(batch.paths))

    for idx, path in enumerate(batch.paths):
        if _batch_cancel.get(batch.batch_id):
            log.info("[batch %s] cancelled before season %d", batch.batch_id, idx + 1)
            batch.status = StepStatus.CANCELLED
            batch.completed_at = datetime.now(timezone.utc).isoformat()
            return

        batch.current_index = idx
        log.info("[batch %s] launching season %d/%d: %s",
                 batch.batch_id, idx + 1, len(batch.paths), path)

        pipeline, err, _code = _launch_pipeline_internal(path, batch.steps, dict(batch.options))
        if err is not None:
            msg = f"season {idx + 1} launch failed: {err.get('error', err)}"
            log.warning("[batch %s] %s", batch.batch_id, msg)
            batch.last_error = msg
            if not batch.continue_on_fail:
                batch.status = StepStatus.FAILED
                batch.completed_at = datetime.now(timezone.utc).isoformat()
                return
            continue

        batch.pipeline_ids.append(pipeline.pipeline_id)
        task = _pipeline_tasks.get(pipeline.pipeline_id)
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                log.info("[batch %s] pipeline %s cancelled",
                         batch.batch_id, pipeline.pipeline_id)
            except Exception as exc:  # _run_pipeline catches its own; defensive
                log.exception("[batch %s] pipeline %s crashed: %s",
                              batch.batch_id, pipeline.pipeline_id, exc)

        final = _pipelines.get(pipeline.pipeline_id)
        final_status = final.status if final else StepStatus.FAILED
        log.info("[batch %s] season %d/%d → %s",
                 batch.batch_id, idx + 1, len(batch.paths), final_status.value)

        if final_status == StepStatus.FAILED:
            batch.last_error = f"season {idx + 1} pipeline failed"
            if not batch.continue_on_fail:
                batch.status = StepStatus.FAILED
                batch.completed_at = datetime.now(timezone.utc).isoformat()
                return

        if _batch_cancel.get(batch.batch_id):
            batch.status = StepStatus.CANCELLED
            batch.completed_at = datetime.now(timezone.utc).isoformat()
            return

    batch.current_index = len(batch.paths)
    batch.status = StepStatus.COMPLETED
    batch.completed_at = datetime.now(timezone.utc).isoformat()
    log.info("[batch %s] completed all %d seasons", batch.batch_id, len(batch.paths))


@router.post("/batch")
async def create_batch(request: Request):
    """Launch a multi-season batch on the server (survives browser close).

    Body::

        {
            "name": "PowerRide -A New Philosophy on Pinning - Craig Jones",
            "paths": ["/data/.../Season_01", "/data/.../Season_02", ...],
            "steps": ["chapters", "subtitles", ...],
            "options": { "mode": "oracle", "force": true },
            "continue_on_fail": true
        }
    """
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
    _batches[batch_id] = batch
    _batch_cancel[batch_id] = False
    _batch_tasks[batch_id] = asyncio.create_task(_run_batch(batch))

    return _serialize_batch(batch)


@router.get("/batch")
async def list_batches(limit: int = 20):
    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    ordered = sorted(_batches.values(), key=lambda b: b.created_at, reverse=True)[:limit]
    return {"batches": [_serialize_batch(b) for b in ordered]}


@router.get("/batch/{batch_id}")
async def get_batch(batch_id: str):
    batch = _batches.get(batch_id)
    if not batch:
        return JSONResponse({"error": "Batch not found"}, status_code=404)
    return _serialize_batch(batch)


@router.post("/batch/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    batch = _batches.get(batch_id)
    if not batch:
        return JSONResponse({"error": "Batch not found"}, status_code=404)
    if batch.status not in (StepStatus.RUNNING, StepStatus.PENDING):
        return JSONResponse(
            {"error": f"Batch is {batch.status.value}, cannot cancel"},
            status_code=409,
        )
    _batch_cancel[batch_id] = True
    if batch.pipeline_ids:
        active_pid = batch.pipeline_ids[-1]
        active = _pipelines.get(active_pid)
        if active and active.status in (StepStatus.RUNNING, StepStatus.PENDING):
            _pipeline_cancel[active_pid] = True
            t = _pipeline_tasks.get(active_pid)
            if t and not t.done():
                t.cancel()
    return {"batch_id": batch_id, "status": "cancelling"}


@router.get("/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get the current state of a pipeline."""
    pipeline = _pipelines.get(pipeline_id)
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
    """SSE endpoint for real-time pipeline progress."""
    pipeline = _pipelines.get(pipeline_id)
    if not pipeline:
        return JSONResponse({"error": "Pipeline not found"}, status_code=404)

    # Each SSE consumer gets its own queue (fan-out). This prevents multiple
    # clients (or a stale retry) from stealing events from each other.
    queue = _subscribe(pipeline_id)

    # Snapshot existing buffer so a reconnecting client gets full history.
    # Replayed events carry their original ``seq`` so the client can dedupe.
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
                # Skip events that were already flushed as part of the replay
                # snapshot (producer emitted them before we subscribed but the
                # broadcast still enqueued them into our fresh subscriber
                # queue in the tiny window between snapshot and subscribe).
                if data.get("seq", 0) and data["seq"] <= replay_max_seq:
                    continue
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("type") in ("pipeline_completed", "pipeline_failed"):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            _unsubscribe(pipeline_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{pipeline_id}/cancel")
async def cancel_pipeline(pipeline_id: str):
    """Request cancellation of a running pipeline."""
    pipeline = _pipelines.get(pipeline_id)
    if not pipeline:
        return JSONResponse({"error": "Pipeline not found"}, status_code=404)
    if pipeline.status not in (StepStatus.RUNNING, StepStatus.PENDING):
        return JSONResponse(
            {"error": f"Pipeline is {pipeline.status.value}, cannot cancel"},
            status_code=409,
        )
    _pipeline_cancel[pipeline_id] = True
    task = _pipeline_tasks.get(pipeline_id)
    if task and not task.done():
        task.cancel()
    return {"pipeline_id": pipeline_id, "status": "cancelling"}


@router.post("/{pipeline_id}/retry")
async def retry_pipeline(pipeline_id: str):
    """Re-run a finished pipeline using its original path/steps/options."""
    src = _pipelines.get(pipeline_id)
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
    _pipelines[new_id] = new_pipe
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(_run_pipeline(new_pipe, queue))
    _pipeline_tasks[new_id] = task
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
    """List pipelines (summary payload, sorted by created_at desc).

    Query params:
        limit   — max entries to return (default 50).
        offset  — pagination offset (default 0).
        status  — optional filter (pending/running/completed/failed/cancelled).

    Response includes an ``X-Total-Count`` header with the *filtered* total.
    Only summary fields are returned per pipeline — for the complete payload
    (step ``diff``, ``message``, timestamps) use ``GET /api/pipeline/{id}``.
    """
    # Clamp
    if limit < 1:
        limit = 1
    if limit > 500:
        limit = 500
    if offset < 0:
        offset = 0

    ordered = sorted(_pipelines.values(), key=lambda p: p.created_at, reverse=True)
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


