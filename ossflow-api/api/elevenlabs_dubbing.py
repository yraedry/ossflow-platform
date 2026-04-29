"""ElevenLabs Dubbing Studio E2E — orchestrator + routes.

The existing ``api/dubbing.py`` proxies to the *local* XTTS backend via
``dubbing-generator``. This module is a completely separate, optional
path: upload the video to ElevenLabs' Dubbing Studio, poll, download the
rendered MP4, and drop it into ``<Season>/elevenlabs/<filename>`` on the
NAS. No SRT, no translator, no local GPU.

Kept isolated so neither pipeline.py nor the XTTS code needs to know it
exists: the frontend hits a dedicated endpoint, we run our own async
task, and we push status into the shared job store so the usual
``/api/jobs/{id}/events`` SSE stream works unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.elevenlabs_dubbing_client import (
    ElevenLabsDubbingClient,
    ElevenLabsDubbingError,
    resolve_output_path,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/elevenlabs-dubbing", tags=["elevenlabs-dubbing"])

# How often we poll ElevenLabs. A full instructional episode (15-20 min)
# takes ElevenLabs 3-8 min to dub; 10 s keeps status snappy without
# hammering the API. The Studio UI itself polls around that interval.
_POLL_INTERVAL_S = 10
_POLL_TIMEOUT_S = 45 * 60  # 45 min ceiling per episode

_TERMINAL_OK = {"dubbed"}
_TERMINAL_FAIL = {"failed", "error"}

# Global serial queue. ElevenLabs Creator plan tolerates concurrent
# jobs but we deliberately serialize here:
#   - predictable credit burn (no parallel run-away on a buggy batch)
#   - easier user mental model (the progress bar is meaningful)
#   - status=queued is reflected in /api/jobs so the UI sees the
#     waiting state instead of a silent spinner
_JOB_SLOT = asyncio.Semaphore(1)

# Empirical observation (2026-04-22): ElevenLabs Studio takes ~25-30% of
# the source video duration to finish dubbing. A 15-min episode dubs in
# ~4 min. We use 0.28 as the median for the ETA bar; if the job takes
# longer the bar holds at 95% until the real "dubbed" status arrives.
_EL_DUB_FACTOR = 0.28

# ETA is chopped into named stages so the UI can show what's happening
# rather than a silent bar. These %s are deliberate and empirical —
# ElevenLabs does transcribe + translate + synthesize + render
# sequentially but won't tell us which stage it's in, so we map the
# elapsed ratio to a stage label.
_STAGES = (
    (0.00, "Subiendo vídeo a ElevenLabs"),
    (0.10, "Transcribiendo audio"),
    (0.25, "Traduciendo a español"),
    (0.55, "Sintetizando voces clonadas"),
    (0.90, "Renderizando MP4 final"),
)


def _stage_for(elapsed_ratio: float) -> str:
    current = _STAGES[0][1]
    for threshold, label in _STAGES:
        if elapsed_ratio >= threshold:
            current = label
    return current


def _probe_duration_seconds(video_path: Path) -> Optional[float]:
    """Return video duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        dur = float(data.get("format", {}).get("duration", 0))
        return dur if dur > 0 else None
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def _job_host():
    """Late import of the app-level job registry.

    Done at call time, not module top, because ``api.app`` imports this
    router. Importing back here at module load would create a cycle.
    """
    from api import app as _app  # noqa: WPS433 — deliberate late import
    return _app


async def _run_elevenlabs_dubbing(
    job_id: str,
    source_path: Path,
    *,
    source_lang: str,
    target_lang: str,
    num_speakers: int,
    watermark: bool,
) -> None:
    """Background task: upload → poll → download → write to NAS."""
    host = _job_host()
    jobs = host._jobs  # noqa: SLF001
    events = host._job_events  # noqa: SLF001
    persist = host._persist_job  # noqa: SLF001
    JobStatus = host.JobStatus  # noqa: N806

    job = jobs.get(job_id)
    if job is None:
        log.error("elevenlabs_dubbing: job %s vanished before start", job_id)
        return

    async def _emit(data: dict) -> None:
        q = events.get(job_id)
        if q:
            await q.put(data)
        if "status" in data:
            persist(job)

    # Announce the queued → running transition cleanly. If the semaphore
    # is taken we surface a "queued" message so the UI knows the job is
    # alive but waiting. Without this the user sees a silent row.
    if _JOB_SLOT.locked():
        await _emit({
            "status": "queued",
            "stage": "Esperando turno (otro dub en curso)",
            "progress": 0,
            "message": "Esperando turno (otro dub en curso)",
        })

    async with _JOB_SLOT:
        try:
            job.status = JobStatus.RUNNING

            # 0. Probe the source video duration → ETA for the UI bar
            video_duration = await asyncio.to_thread(_probe_duration_seconds, source_path)
            estimated_total = (
                max(60, video_duration * _EL_DUB_FACTOR) if video_duration else 4 * 60
            )
            job.result = {
                "provider": "elevenlabs",
                "video_duration_sec": video_duration,
                "estimated_total_sec": estimated_total,
            }
            await _emit({
                "status": "running",
                "stage": _stage_for(0.0),
                "progress": 0,
                "elapsed_sec": 0,
                "estimated_total_sec": int(estimated_total),
                "estimated_remaining_sec": int(estimated_total),
                "message": _stage_for(0.0),
            })

            # 1. Start job (blocking I/O → thread)
            started_at = time.monotonic()

            def _start() -> tuple[str, str]:
                client = ElevenLabsDubbingClient()
                with source_path.open("rb") as f:
                    dj = client.start(
                        file=f,
                        filename=source_path.name,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        num_speakers=num_speakers,
                        watermark=watermark,
                        name=source_path.stem,
                    )
                return dj.dubbing_id, dj.status

            dubbing_id, initial_status = await asyncio.to_thread(_start)
            job.result = {**job.result, "dubbing_id": dubbing_id}
            # CRITICAL: flush to disk now. If the container dies before
            # the first poll arrives we still need the dubbing_id to
            # resume on startup. The generic ``_emit`` only flushes on
            # status changes, so we call persist directly here.
            persist(job)
            log.info(
                "elevenlabs_dubbing: job=%s started id=%s video_dur=%.1fs est_total=%.1fs",
                job_id, dubbing_id, video_duration or -1, estimated_total,
            )

            # 2. Poll until terminal. Progress bar is deterministic on
            #    ``elapsed / estimated_total`` but capped at 95% so it never
            #    claims "done" before ElevenLabs reports status=dubbed.
            def _poll_once() -> str:
                client = ElevenLabsDubbingClient()
                return client.poll(dubbing_id).status

            status = initial_status
            while status not in _TERMINAL_OK | _TERMINAL_FAIL:
                await asyncio.sleep(_POLL_INTERVAL_S)
                elapsed = time.monotonic() - started_at
                if elapsed > _POLL_TIMEOUT_S:
                    raise ElevenLabsDubbingError(
                        f"timeout after {_POLL_TIMEOUT_S}s waiting for {dubbing_id}"
                    )
                status = await asyncio.to_thread(_poll_once)
                ratio = min(0.95, elapsed / estimated_total)
                pct = int(ratio * 100)
                remaining = max(0, int(estimated_total - elapsed))
                stage = _stage_for(ratio)
                job.progress = pct
                job.message = stage
                await _emit({
                    "progress": pct,
                    "stage": stage,
                    "elapsed_sec": int(elapsed),
                    "estimated_total_sec": int(estimated_total),
                    "estimated_remaining_sec": remaining,
                    "message": stage,
                })

            if status in _TERMINAL_FAIL:
                raise ElevenLabsDubbingError(f"ElevenLabs reported status={status}")

            # 3. Download
            await _emit({
                "progress": 96,
                "stage": "Descargando MP4 doblado",
                "message": "Descargando MP4 doblado",
            })

            def _download() -> bytes:
                client = ElevenLabsDubbingClient()
                return client.download(dubbing_id, target_lang)

            data = await asyncio.to_thread(_download)

            # 4. Write to <Season>/elevenlabs/<nombre_original>
            output_path = resolve_output_path(source_path)

            def _write() -> int:
                output_path.write_bytes(data)
                return output_path.stat().st_size

            size = await asyncio.to_thread(_write)
            total_elapsed = int(time.monotonic() - started_at)
            job.result = {
                **job.result,
                "output_path": str(output_path),
                "output_filename": output_path.name,
                "bytes": size,
                "total_elapsed_sec": total_elapsed,
            }

            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.completed_at = datetime.now().isoformat()
            job.message = f"Listo: {output_path.name}"
            await _emit({
                "status": "completed",
                "progress": 100,
                "stage": "Completado",
                "elapsed_sec": total_elapsed,
                "estimated_remaining_sec": 0,
                "result": job.result,
                "message": f"Guardado en {output_path}",
            })
            log.info(
                "elevenlabs_dubbing: job=%s wrote %d bytes to %s (took %ds)",
                job_id, size, output_path, total_elapsed,
            )

        except Exception as exc:  # noqa: BLE001 — every error reaches the UI
            job.status = JobStatus.FAILED
            job.message = str(exc)
            job.completed_at = datetime.now().isoformat()
            log.exception("elevenlabs_dubbing: job=%s failed", job_id)
            await _emit({"status": "failed", "message": str(exc)})


async def _resume_elevenlabs_dubbing(
    job_id: str,
    source_path: Path,
    dubbing_id: str,
    *,
    target_lang: str = "es",
) -> None:
    """Re-attach to a dubbing job already running on ElevenLabs.

    Called on startup when we find a persisted job in ``running`` state
    with a ``dubbing_id`` but no asyncio task alive (e.g. the container
    was rebuilt mid-flight). We skip the upload stage and jump straight
    into polling — ElevenLabs kept the job going, we just need to pick
    the result up.
    """
    host = _job_host()
    jobs = host._jobs  # noqa: SLF001
    events = host._job_events  # noqa: SLF001
    persist = host._persist_job  # noqa: SLF001
    JobStatus = host.JobStatus  # noqa: N806

    job = jobs.get(job_id)
    if job is None:
        log.error("elevenlabs_dubbing: resume of %s failed — job vanished", job_id)
        return

    # Make sure an event queue exists for this job even after restart,
    # otherwise the SSE endpoint would have to create one on demand and
    # miss early progress events.
    if job_id not in events:
        events[job_id] = asyncio.Queue()

    async def _emit(data: dict) -> None:
        q = events.get(job_id)
        if q:
            await q.put(data)
        if "status" in data:
            persist(job)

    if _JOB_SLOT.locked():
        await _emit({
            "status": "queued",
            "stage": "Esperando turno (reanudado)",
            "progress": 0,
            "message": "Esperando turno (reanudado)",
        })

    async with _JOB_SLOT:
        try:
            job.status = JobStatus.RUNNING
            estimated_total = (
                (job.result or {}).get("estimated_total_sec") or 4 * 60
            )
            started_at = time.monotonic()
            await _emit({
                "status": "running",
                "stage": "Reanudando poll (recuperado tras reinicio)",
                "progress": max(0, int(job.progress or 0)),
                "elapsed_sec": 0,
                "estimated_total_sec": int(estimated_total),
                "estimated_remaining_sec": int(estimated_total),
                "message": "Reanudando tras reinicio del contenedor",
            })
            log.info(
                "elevenlabs_dubbing: resuming job=%s dubbing_id=%s",
                job_id, dubbing_id,
            )

            def _poll_once() -> str:
                client = ElevenLabsDubbingClient()
                return client.poll(dubbing_id).status

            status = await asyncio.to_thread(_poll_once)
            while status not in _TERMINAL_OK | _TERMINAL_FAIL:
                await asyncio.sleep(_POLL_INTERVAL_S)
                elapsed = time.monotonic() - started_at
                if elapsed > _POLL_TIMEOUT_S:
                    raise ElevenLabsDubbingError(
                        f"timeout after resume waiting for {dubbing_id}"
                    )
                status = await asyncio.to_thread(_poll_once)
                ratio = min(0.95, elapsed / estimated_total)
                pct = int(ratio * 100)
                remaining = max(0, int(estimated_total - elapsed))
                stage = _stage_for(ratio)
                job.progress = pct
                job.message = stage
                await _emit({
                    "progress": pct,
                    "stage": stage,
                    "elapsed_sec": int(elapsed),
                    "estimated_total_sec": int(estimated_total),
                    "estimated_remaining_sec": remaining,
                    "message": stage,
                })

            if status in _TERMINAL_FAIL:
                raise ElevenLabsDubbingError(f"ElevenLabs reported status={status}")

            await _emit({
                "progress": 96,
                "stage": "Descargando MP4 doblado",
                "message": "Descargando MP4 doblado",
            })

            def _download() -> bytes:
                client = ElevenLabsDubbingClient()
                return client.download(dubbing_id, target_lang)

            data = await asyncio.to_thread(_download)
            output_path = resolve_output_path(source_path)

            def _write() -> int:
                output_path.write_bytes(data)
                return output_path.stat().st_size

            size = await asyncio.to_thread(_write)
            total_elapsed = int(time.monotonic() - started_at)
            job.result = {
                **(job.result or {}),
                "output_path": str(output_path),
                "output_filename": output_path.name,
                "bytes": size,
                "total_elapsed_sec": total_elapsed,
                "resumed": True,
            }

            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.completed_at = datetime.now().isoformat()
            job.message = f"Listo (reanudado): {output_path.name}"
            await _emit({
                "status": "completed",
                "progress": 100,
                "stage": "Completado",
                "elapsed_sec": total_elapsed,
                "estimated_remaining_sec": 0,
                "result": job.result,
                "message": f"Guardado en {output_path}",
            })
            log.info(
                "elevenlabs_dubbing: resumed job=%s wrote %d bytes to %s",
                job_id, size, output_path,
            )

        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.message = f"resume failed: {exc}"
            job.completed_at = datetime.now().isoformat()
            log.exception("elevenlabs_dubbing: resume of job=%s failed", job_id)
            await _emit({"status": "failed", "message": job.message})


def resume_orphan_jobs() -> dict:
    """Scan persisted jobs and re-attach or fail any orphans.

    Call this from the FastAPI lifespan startup hook *after*
    ``_load_persisted_jobs`` has rehydrated ``_jobs`` from disk.

    Returns a summary ``{"resumed": [...], "failed": [...]}`` so callers
    can log what happened. Side effects:
      - jobs with dubbing_id → asyncio task scheduled to resume
      - jobs without dubbing_id → marked FAILED (we don't know what to poll)
    """
    host = _job_host()
    JobStatus = host.JobStatus  # noqa: N806
    resumed: list[str] = []
    failed: list[str] = []

    for job_id, job in list(host._jobs.items()):  # noqa: SLF001
        if job.job_type != "elevenlabs_dubbing":
            continue
        status = job.status.value if hasattr(job.status, "value") else str(job.status)
        if status not in ("running", "queued"):
            continue
        result = job.result or {}
        dubbing_id = result.get("dubbing_id")
        source_path = Path(job.video_path) if job.video_path else None

        if not dubbing_id or source_path is None or not source_path.exists():
            job.status = JobStatus.FAILED
            job.message = (
                "lost on container restart (no dubbing_id recorded yet)"
                if not dubbing_id
                else f"source video missing: {job.video_path}"
            )
            job.completed_at = datetime.now().isoformat()
            host._persist_job(job)  # noqa: SLF001
            failed.append(job_id)
            continue

        # Rehydrate event queue and spawn resume task
        host._job_events.setdefault(job_id, asyncio.Queue())  # noqa: SLF001
        asyncio.create_task(
            _resume_elevenlabs_dubbing(job_id, source_path, dubbing_id)
        )
        resumed.append(job_id)

    if resumed or failed:
        log.info(
            "elevenlabs_dubbing: startup resume → resumed=%s failed=%s",
            resumed, failed,
        )
    return {"resumed": resumed, "failed": failed}


# Minimum plausible size for a "real" dubbed output. Anything smaller is
# treated as a stale/corrupt artefact and overwritten instead of skipped.
# Empirically the smallest valid ElevenLabs mp4 we've seen for a 2-min
# clip is ~30 MB; 1 MB is a very conservative lower bound.
_SKIP_MIN_BYTES = 1 * 1024 * 1024


def _already_dubbed(source_video: Path) -> bool:
    """True if an ElevenLabs dub already exists for this video.

    Used as a defensive skip for batch runs: if the user re-launches a
    batch that was interrupted halfway, we don't re-burn credits on the
    videos that already landed in ``<Season>/elevenlabs/``. Single-video
    endpoint does NOT skip — the user may want to intentionally regenerate.
    """
    out = source_video.parent / "elevenlabs" / source_video.name
    if not out.exists() or not out.is_file():
        return False
    try:
        return out.stat().st_size >= _SKIP_MIN_BYTES
    except OSError:
        return False


def _spawn_job(
    source: Path,
    *,
    source_lang: str,
    target_lang: str,
    num_speakers: int,
    watermark: bool,
) -> str:
    """Register a new job + background task. Returns job_id.

    Shared helper between the single-video and batch endpoints so the
    registration logic lives in one place.
    """
    host = _job_host()
    job_id = str(uuid.uuid4())[:8]
    job = host.JobInfo(
        job_id=job_id,
        job_type="elevenlabs_dubbing",
        video_path=str(source),
    )
    host._jobs[job_id] = job  # noqa: SLF001
    host._job_events[job_id] = asyncio.Queue()  # noqa: SLF001
    host._persist_job(job)  # noqa: SLF001

    asyncio.create_task(_run_elevenlabs_dubbing(
        job_id,
        source,
        source_lang=source_lang,
        target_lang=target_lang,
        num_speakers=num_speakers,
        watermark=watermark,
    ))
    return job_id


_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}


@router.post("")
async def create_elevenlabs_dubbing_job(request: Request) -> dict:
    """Kick off an ElevenLabs Dubbing Studio job for a single video.

    Request body::

        {
          "path": "/media/.../S01E02 - Foo.mp4",  # REQUIRED
          "source_lang": "en",                     # optional, default "en"
          "target_lang": "es",                     # optional, default "es"
          "num_speakers": 1,                       # optional, default 1
          "watermark": true                        # optional, default true
        }

    Returns ``{"job_id": "...", "status": "queued"}``. Track via
    ``GET /api/jobs/{job_id}`` and ``GET /api/jobs/{job_id}/events``.
    """
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    path = body.get("path")
    if not path or not isinstance(path, str):
        raise HTTPException(status_code=400, detail="'path' is required")
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {path}")

    source_lang = str(body.get("source_lang") or "en").strip() or "en"
    target_lang = str(body.get("target_lang") or "es").strip() or "es"
    num_speakers = int(body.get("num_speakers") or 1)
    watermark = bool(body.get("watermark", True))

    job_id = _spawn_job(
        source,
        source_lang=source_lang,
        target_lang=target_lang,
        num_speakers=num_speakers,
        watermark=watermark,
    )
    return {"job_id": job_id, "status": "queued"}


@router.post("/batch")
async def create_elevenlabs_dubbing_batch(request: Request) -> dict:
    """Queue ElevenLabs dubbing jobs for every video in a season folder.

    Request body::

        {
          "season_path": "/media/.../Season 01",  # REQUIRED
          "source_lang": "en",
          "target_lang": "es",
          "num_speakers": 1,
          "watermark": true
        }

    Videos already present as ``<season>/elevenlabs/<filename>`` with a
    plausible size are skipped to avoid burning credits when a
    previously-interrupted batch is re-launched. Jobs are globally
    serialized (one active at a time) via ``_JOB_SLOT``.
    """
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    season_path = body.get("season_path")
    if not season_path or not isinstance(season_path, str):
        raise HTTPException(status_code=400, detail="'season_path' is required")
    season = Path(season_path)
    if not season.exists() or not season.is_dir():
        raise HTTPException(status_code=404, detail=f"Season folder not found: {season_path}")

    source_lang = str(body.get("source_lang") or "en").strip() or "en"
    target_lang = str(body.get("target_lang") or "es").strip() or "es"
    num_speakers = int(body.get("num_speakers") or 1)
    watermark = bool(body.get("watermark", True))

    queued: list[dict] = []
    skipped: list[dict] = []
    for child in sorted(season.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_file() or child.suffix.lower() not in _VIDEO_EXTS:
            continue
        # Skip the dubbed outputs from the legacy XTTS pipeline too —
        # those carry the "_DOBLADO" suffix and are artefacts, not sources.
        if child.stem.endswith("_DOBLADO"):
            continue
        if _already_dubbed(child):
            skipped.append({"filename": child.name, "reason": "already_dubbed"})
            continue
        job_id = _spawn_job(
            child,
            source_lang=source_lang,
            target_lang=target_lang,
            num_speakers=num_speakers,
            watermark=watermark,
        )
        queued.append({"filename": child.name, "job_id": job_id})

    return {
        "season_path": str(season),
        "queued": queued,
        "skipped": skipped,
        "queued_count": len(queued),
        "skipped_count": len(skipped),
    }


@router.get("")
async def list_elevenlabs_dubbing_jobs(limit: int = 50) -> dict:
    """List ElevenLabs dubbing jobs (active + last ``limit`` completed).

    Active jobs come first, sorted by created_at desc; completed/failed
    jobs follow, capped at ``limit`` to keep the response bounded.
    """
    host = _job_host()
    all_jobs = [j for j in host._jobs.values() if j.job_type == "elevenlabs_dubbing"]  # noqa: SLF001
    active_states = {"queued", "running"}
    active = [j for j in all_jobs if j.status.value in active_states]
    finished = [j for j in all_jobs if j.status.value not in active_states]
    finished.sort(key=lambda j: j.completed_at or "", reverse=True)
    return {
        "active": [asdict(j) for j in active],
        "recent": [asdict(j) for j in finished[:limit]],
    }


@router.get("/{job_id}")
async def get_elevenlabs_dubbing_status(job_id: str):
    """Alias over the app-level job registry for convenience.

    Frontends already polling /api/jobs/{id} don't need this, but a
    provider-scoped endpoint is handy for admin scripts.
    """
    host = _job_host()
    job = host._jobs.get(job_id)  # noqa: SLF001
    if not job or job.job_type != "elevenlabs_dubbing":
        return JSONResponse({"error": "not found"}, status_code=404)
    return asdict(job)
