"""Servicio del doblaje ElevenLabs Studio.

**Acoplamiento residual (rotura parcial #7):** este servicio sigue
accediendo al registry global de jobs vía ``api.app`` con late import. La
rotura completa se hace en F4.T19 cuando se cree ``JobsService``: en ese
momento el servicio recibirá ``JobsService`` por DI y desaparecerán los
``_job_host()``.

Acoplamiento #6 cerrado: ``resume_orphan_jobs`` ahora es un símbolo
público que ``infrastructure.lifespan`` registra como hook de startup
sin importar privados.
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

from ossflow_api.clients.elevenlabs import (
    ElevenLabsDubbingClient,
    ElevenLabsDubbingError,
    resolve_output_path,
)

log = logging.getLogger(__name__)

# Polling cadence. Un episodio típico de instructional (15-20 min) tarda
# 3-8 min en doblarse; 10s mantiene el status fresco sin saturar la API.
_POLL_INTERVAL_S = 10
_POLL_TIMEOUT_S = 45 * 60  # 45 min ceiling per episode

_TERMINAL_OK = {"dubbed"}
_TERMINAL_FAIL = {"failed", "error"}

# Cola serial global. ElevenLabs Creator plan tolera concurrencia, pero
# serializamos:
#   - quema de créditos predecible
#   - mental model más simple para el usuario
#   - status=queued aparece en /api/jobs (UI ve el waiting)
_JOB_SLOT = asyncio.Semaphore(1)

# Observación empírica (2026-04-22): ElevenLabs Studio tarda ~25-30% de
# la duración del vídeo origen. 0.28 es la mediana del ETA bar.
_EL_DUB_FACTOR = 0.28

# El ETA está dividido en stages para que la UI muestre algo más útil que
# una barra silente. Los % son empíricos.
_STAGES = (
    (0.00, "Subiendo vídeo a ElevenLabs"),
    (0.10, "Transcribiendo audio"),
    (0.25, "Traduciendo a español"),
    (0.55, "Sintetizando voces clonadas"),
    (0.90, "Renderizando MP4 final"),
)

# Tamaño mínimo plausible para un dub válido. Por debajo, lo tratamos
# como artefacto stale.
_SKIP_MIN_BYTES = 1 * 1024 * 1024


def _stage_for(elapsed_ratio: float) -> str:
    current = _STAGES[0][1]
    for threshold, label in _STAGES:
        if elapsed_ratio >= threshold:
            current = label
    return current


def _probe_duration_seconds(video_path: Path) -> Optional[float]:
    """Devuelve la duración del vídeo via ffprobe, o None si falla."""
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
    """Late import del registry de jobs a nivel app.

    A nivel de import time crearíamos un ciclo (api.app importa este
    módulo). Con late import en cada call rompemos el ciclo.

    Cuando F4.T19 cree ``JobsService``, esta función desaparece y los
    callers reciben ``JobsService`` por DI.
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
            # CRITICAL: flush a disco YA. Si el contenedor cae antes del
            # primer poll necesitamos el dubbing_id para resumir al
            # arranque. _emit solo flushea en cambios de status.
            persist(job)
            log.info(
                "elevenlabs_dubbing: job=%s started id=%s video_dur=%.1fs est_total=%.1fs",
                job_id, dubbing_id, video_duration or -1, estimated_total,
            )

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

        except Exception as exc:  # noqa: BLE001
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
    """Reengancha a un dubbing que ya estaba corriendo en ElevenLabs."""
    host = _job_host()
    jobs = host._jobs  # noqa: SLF001
    events = host._job_events  # noqa: SLF001
    persist = host._persist_job  # noqa: SLF001
    JobStatus = host.JobStatus  # noqa: N806

    job = jobs.get(job_id)
    if job is None:
        log.error("elevenlabs_dubbing: resume of %s failed — job vanished", job_id)
        return

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
    """Re-engancha jobs persistidos que estaban en vuelo al reiniciar.

    Llamar desde el lifespan startup *después* de ``_load_persisted_jobs``.
    Acoplamiento #6 cerrado: este símbolo es público y se registra como
    hook sin tocar privados.
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


def already_dubbed(source_video: Path) -> bool:
    """True si ya existe un dub válido para este vídeo."""
    out = source_video.parent / "elevenlabs" / source_video.name
    if not out.exists() or not out.is_file():
        return False
    try:
        return out.stat().st_size >= _SKIP_MIN_BYTES
    except OSError:
        return False


def spawn_job(
    source: Path,
    *,
    source_lang: str,
    target_lang: str,
    num_speakers: int,
    watermark: bool,
) -> str:
    """Registra un nuevo job + background task. Devuelve job_id."""
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


def list_jobs(limit: int = 50) -> dict:
    """Lista los jobs ElevenLabs activos + últimos completados."""
    host = _job_host()
    all_jobs = [
        j for j in host._jobs.values()  # noqa: SLF001
        if j.job_type == "elevenlabs_dubbing"
    ]
    active_states = {"queued", "running"}
    active = [j for j in all_jobs if j.status.value in active_states]
    finished = [j for j in all_jobs if j.status.value not in active_states]
    finished.sort(key=lambda j: j.completed_at or "", reverse=True)
    return {
        "active": [asdict(j) for j in active],
        "recent": [asdict(j) for j in finished[:limit]],
    }


def get_job(job_id: str) -> Optional[dict]:
    """Devuelve el dict del job o None si no existe."""
    host = _job_host()
    job = host._jobs.get(job_id)  # noqa: SLF001
    if not job or job.job_type != "elevenlabs_dubbing":
        return None
    return asdict(job)
