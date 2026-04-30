"""Repositorio SQL del sistema "background jobs" (tabla ``background_jobs``).

Refactor del antiguo ``api.background_jobs.JobRegistry._load/_save/...``,
acotado a la responsabilidad única de **acceso a datos**. Lanzar tareas
en background lo hace ``services.background.BackgroundJobsService`` con
``_internal.scheduler.JobsScheduler``.

Este repositorio solo expone primitivas CRUD + recovery de huérfanos. No
sabe de asyncio, threading ni FastAPI.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ossflow_service_kit.db import init_db, session_scope
from ossflow_service_kit.db.models import BackgroundJob as BackgroundJobRow

from ..models import BackgroundJob, JobStatus
from .._internal.orphan_recovery import mark_running_as_failed

log = logging.getLogger(__name__)

MAX_ENTRIES = 100


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _row_to_dataclass(row: BackgroundJobRow) -> BackgroundJob:
    """Mapea fila SQL → ``BackgroundJob`` dataclass."""
    try:
        payload = json.loads(row.payload) if row.payload else {}
    except json.JSONDecodeError:
        payload = {}
    return BackgroundJob(
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


def _serialize_payload(job: BackgroundJob) -> str:
    """Convierte ``progress``, ``message`` y ``params`` en una cadena JSON
    única que se persiste en la columna ``payload``."""
    return json.dumps(
        {
            "progress": job.progress,
            "message": job.message,
            "params": job.params,
        },
        ensure_ascii=False,
    )


class BackgroundJobsRepository:
    """Acceso CRUD a la tabla ``background_jobs``."""

    def __init__(self, *, history_file: Optional[Path] = None) -> None:
        # Fichero legacy ``CONFIG_DIR/background_jobs.json``. Se importa una
        # sola vez al primer arranque y se renombra a ``.bak``. Idempotente.
        self._history_file = history_file

    # --- arranque ---------------------------------------------------------

    def init_db_and_recover(self) -> int:
        """Crea tablas (idempotente), importa legacy JSON una vez, y marca
        jobs ``RUNNING``/``QUEUED`` como ``FAILED``.

        Devuelve el número de huérfanos recuperados.
        """
        try:
            init_db()
        except Exception as exc:  # noqa: BLE001
            log.warning("init_db failed: %s", exc)
            return 0
        self._import_legacy_once()
        return self._recover_orphans()

    def _import_legacy_once(self) -> None:
        if not self._history_file or not self._history_file.exists():
            return
        try:
            raw = json.loads(self._history_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
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
                    s.add(BackgroundJobRow(
                        id=d["id"],
                        type=d.get("type", "unknown"),
                        status=d.get("status", JobStatus.FAILED.value),
                        payload=json.dumps(payload, ensure_ascii=False),
                        result=json.dumps(d["result"]) if d.get("result") else None,
                        error=d.get("error"),
                        created_at=_parse_dt(d.get("created_at")),
                        finished_at=_parse_dt(d.get("completed_at")),
                    ))
            backup = self._history_file.with_suffix(".json.bak")
            self._history_file.rename(backup)
            log.info("Imported legacy background_jobs.json → DB (backup %s)", backup)
        except Exception as exc:  # noqa: BLE001
            log.warning("Legacy background_jobs import failed: %s", exc)

    def _recover_orphans(self) -> int:
        """Marca como FAILED los jobs que estaban RUNNING/QUEUED al arranque."""
        try:
            with session_scope() as s:
                rows = s.query(BackgroundJobRow).filter(
                    BackgroundJobRow.status.in_(
                        [JobStatus.RUNNING.value, JobStatus.QUEUED.value]
                    )
                ).all()

                def _setter(row: BackgroundJobRow, status: str, error: str, completed_at: str) -> None:
                    row.status = status
                    row.error = row.error or error
                    if not row.finished_at:
                        row.finished_at = datetime.fromisoformat(completed_at)

                return mark_running_as_failed(rows, _setter)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to recover orphan background jobs: %s", exc)
            return 0

    # --- CRUD -------------------------------------------------------------

    def get(self, job_id: str) -> Optional[BackgroundJob]:
        try:
            with session_scope() as s:
                row = s.get(BackgroundJobRow, job_id)
                return _row_to_dataclass(row) if row else None
        except Exception as exc:  # noqa: BLE001
            log.warning("get(%s) failed: %s", job_id, exc)
            return None

    def list_all(self, type_filter: Optional[str] = None) -> list[BackgroundJob]:
        """Listado ordenado por ``created_at`` desc, trimeado a ``MAX_ENTRIES``."""
        try:
            with session_scope() as s:
                q = s.query(BackgroundJobRow)
                if type_filter:
                    q = q.filter(BackgroundJobRow.type == type_filter)
                q = q.order_by(BackgroundJobRow.created_at.desc()).limit(MAX_ENTRIES)
                return [_row_to_dataclass(row) for row in q.all()]
        except Exception as exc:  # noqa: BLE001
            log.warning("list_all failed: %s", exc)
            return []

    def upsert(self, job: BackgroundJob) -> None:
        """Crea o actualiza un job. ``self.trim_to(MAX_ENTRIES)`` se llama
        después implícitamente por el caller."""
        try:
            with session_scope() as s:
                created = _parse_dt(job.created_at) or datetime.now()
                finished = _parse_dt(job.completed_at)
                row = s.get(BackgroundJobRow, job.id)
                if row is None:
                    s.add(BackgroundJobRow(
                        id=job.id,
                        type=job.type,
                        status=job.status,
                        payload=_serialize_payload(job),
                        result=json.dumps(job.result) if job.result else None,
                        error=job.error,
                        created_at=created,
                        finished_at=finished,
                    ))
                else:
                    row.type = job.type
                    row.status = job.status
                    row.payload = _serialize_payload(job)
                    row.result = json.dumps(job.result) if job.result else None
                    row.error = job.error
                    row.finished_at = finished
        except Exception as exc:  # noqa: BLE001
            log.warning("upsert(%s) failed: %s", job.id, exc)

    def trim_to(self, max_entries: int) -> None:
        """Elimina filas más allá de las ``max_entries`` más recientes."""
        try:
            with session_scope() as s:
                # IDs a conservar (las N más recientes).
                keep = {
                    row.id for row in s.query(BackgroundJobRow.id).order_by(
                        BackgroundJobRow.created_at.desc()
                    ).limit(max_entries).all()
                }
                # Borrar el resto.
                for row in s.query(BackgroundJobRow).all():
                    if row.id not in keep:
                        s.delete(row)
        except Exception as exc:  # noqa: BLE001
            log.warning("trim_to failed: %s", exc)
