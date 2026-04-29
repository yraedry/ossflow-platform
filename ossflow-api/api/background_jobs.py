"""Generic persistent async background-job registry.

Used for long-running scans (cleanup, duplicates) so the user can fire them
and leave the page while they run in the background.

Shape of a serialized job::

    {
      "id": "abc123",
      "type": "cleanup_scan",
      "status": "running",
      "progress": 42.0,
      "message": "Scanning 1200 files...",
      "result": null,
      "error": null,
      "created_at": "...",
      "completed_at": null,
      "params": {"path": "/media/X"}
    }

Jobs that were ``running`` when the server died are marked ``failed`` on
import (same pattern used in ``api.pipeline`` for pipeline history).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, HTTPException

from api.settings import CONFIG_DIR as _CONFIG_DIR

from bjj_service_kit.db import init_db, session_scope
from bjj_service_kit.db.models import BackgroundJob as BackgroundJobRow

log = logging.getLogger(__name__)

HISTORY_FILE = _CONFIG_DIR / "background_jobs.json"  # legacy, used for one-time import
MAX_ENTRIES = 100

# Job statuses
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"

ProgressCallback = Callable[[Optional[float], str], None]
CoroFactory = Callable[[ProgressCallback], Awaitable[dict]]


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


@dataclass
class BackgroundJob:
    id: str
    type: str
    status: str = QUEUED
    progress: Optional[float] = None
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class JobRegistry:
    """In-memory registry of background jobs with JSON persistence."""

    def __init__(self, history_file: Path = HISTORY_FILE) -> None:
        self._jobs: dict[str, BackgroundJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._threads: dict[str, threading.Thread] = {}
        self.history_file = history_file
        self._lock = threading.Lock()
        self._load()

    # --- persistence -----------------------------------------------------
    def _load(self) -> None:
        try:
            init_db()
        except Exception as exc:
            log.warning("init_db failed: %s", exc)
            return
        self._import_legacy_once()
        try:
            with session_scope() as s:
                rows = s.query(BackgroundJobRow).all()
                for row in rows:
                    try:
                        payload = json.loads(row.payload) if row.payload else {}
                    except json.JSONDecodeError:
                        payload = {}
                    job = BackgroundJob(
                        id=row.id,
                        type=row.type,
                        status=row.status,
                        progress=payload.get("progress"),
                        message=payload.get("message", ""),
                        result=json.loads(row.result) if row.result else None,
                        error=row.error,
                        created_at=row.created_at.isoformat() if row.created_at else datetime.now().isoformat(),
                        completed_at=row.finished_at.isoformat() if row.finished_at else None,
                        params=payload.get("params", {}),
                    )
                    if job.status in (RUNNING, QUEUED):
                        job.status = FAILED
                        job.error = job.error or "interrupted: server restarted"
                        job.completed_at = job.completed_at or datetime.now().isoformat()
                        # Persist the fix
                        row.status = FAILED
                        row.error = job.error
                        if not row.finished_at:
                            row.finished_at = datetime.now()
                    self._jobs[job.id] = job
        except Exception as exc:
            log.warning("Failed to load background jobs from DB: %s", exc)

    def _import_legacy_once(self) -> None:
        if not self.history_file.exists():
            return
        try:
            raw = json.loads(self.history_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, list):
            return
        try:
            with session_scope() as s:
                existing = {r.id for r in s.query(BackgroundJobRow.id).all()}
                for d in raw:
                    if not isinstance(d, dict) or d.get("id") in existing:
                        continue
                    payload = {
                        "progress": d.get("progress"),
                        "message": d.get("message", ""),
                        "params": d.get("params", {}),
                    }
                    created = _parse_dt(d.get("created_at"))
                    finished = _parse_dt(d.get("completed_at"))
                    s.add(BackgroundJobRow(
                        id=d["id"],
                        type=d.get("type", "unknown"),
                        status=d.get("status", FAILED),
                        payload=json.dumps(payload, ensure_ascii=False),
                        result=json.dumps(d["result"]) if d.get("result") else None,
                        error=d.get("error"),
                        created_at=created,
                        finished_at=finished,
                    ))
            backup = self.history_file.with_suffix(".json.bak")
            self.history_file.rename(backup)
            log.info("Imported legacy background_jobs.json → DB (backup %s)", backup)
        except Exception as exc:
            log.warning("Legacy background_jobs import failed: %s", exc)

    def _save(self) -> None:
        """Persist + trim. Called after any job mutation."""
        try:
            items = sorted(
                self._jobs.values(), key=lambda j: j.created_at, reverse=True
            )[:MAX_ENTRIES]
            self._jobs = {j.id: j for j in items}
            keep_ids = set(self._jobs.keys())
            with session_scope() as s:
                # Upsert known jobs
                for job in self._jobs.values():
                    payload = {
                        "progress": job.progress,
                        "message": job.message,
                        "params": job.params,
                    }
                    created = _parse_dt(job.created_at) or datetime.now()
                    finished = _parse_dt(job.completed_at)
                    row = s.get(BackgroundJobRow, job.id)
                    if row is None:
                        s.add(BackgroundJobRow(
                            id=job.id,
                            type=job.type,
                            status=job.status,
                            payload=json.dumps(payload, ensure_ascii=False),
                            result=json.dumps(job.result) if job.result else None,
                            error=job.error,
                            created_at=created,
                            finished_at=finished,
                        ))
                    else:
                        row.type = job.type
                        row.status = job.status
                        row.payload = json.dumps(payload, ensure_ascii=False)
                        row.result = json.dumps(job.result) if job.result else None
                        row.error = job.error
                        row.finished_at = finished
                # Trim excess rows (keep same 100)
                for row in s.query(BackgroundJobRow).all():
                    if row.id not in keep_ids:
                        s.delete(row)
        except Exception as exc:
            log.warning("Failed to persist background jobs: %s", exc)

    # --- public API ------------------------------------------------------
    def get(self, job_id: str) -> Optional[BackgroundJob]:
        return self._jobs.get(job_id)

    def list_all(self, type_filter: Optional[str] = None) -> list[BackgroundJob]:
        items = sorted(
            self._jobs.values(), key=lambda j: j.created_at, reverse=True
        )
        if type_filter:
            items = [j for j in items if j.type == type_filter]
        return items

    def submit(
        self,
        type: str,
        coro_factory: CoroFactory,
        params: dict,
    ) -> BackgroundJob:
        """Schedule ``coro_factory`` as an asyncio task.

        ``coro_factory`` receives a ``update_progress(percent, message)``
        callback. Its return value (``dict``) is stored in ``job.result``.
        """
        job_id = uuid.uuid4().hex[:12]
        job = BackgroundJob(id=job_id, type=type, params=dict(params or {}))
        self._jobs[job_id] = job
        self._save()

        def update_progress(percent: Optional[float], message: str = "") -> None:
            if percent is not None:
                try:
                    job.progress = float(percent)
                except (TypeError, ValueError):
                    pass
            if message:
                job.message = message

        async def _runner() -> None:
            job.status = RUNNING
            self._save()
            try:
                result = await coro_factory(update_progress)
                job.result = result if isinstance(result, dict) else {"value": result}
                job.status = COMPLETED
                job.progress = 100.0
            except Exception as exc:  # noqa: BLE001
                log.exception("Background job %s failed", job_id)
                job.status = FAILED
                job.error = f"{exc.__class__.__name__}: {exc}"
            finally:
                job.completed_at = datetime.now().isoformat()
                self._save()

        def _thread_target() -> None:
            # Own event loop per job so the task is not tied to the request's
            # lifecycle (TestClient tears down its loop after each call).
            try:
                asyncio.run(_runner())
            finally:
                self._threads.pop(job_id, None)

        try:
            t = threading.Thread(
                target=_thread_target,
                name=f"bgjob-{job_id}",
                daemon=True,
            )
            self._threads[job_id] = t
            t.start()
        except RuntimeError as exc:
            job.status = FAILED
            job.error = f"scheduling failed: {exc}"
            job.completed_at = datetime.now().isoformat()
            self._save()
        return job


# Singleton
registry = JobRegistry()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/background-jobs", tags=["background"])


@router.get("")
@router.get("/")
async def list_jobs(type: Optional[str] = None) -> dict:
    return {"jobs": [j.to_dict() for j in registry.list_all(type_filter=type)]}


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()
