"""Repositorio SQL del sistema "legacy jobs" (tabla ``legacy_jobs``).

Refactor del antiguo ``api.jobs_store.JobsStore`` + estado global de
``api.app._jobs``. Acotado a la responsabilidad única de **acceso a
datos**. La gestión de SSE y scheduling vive en
``services.legacy.LegacyJobsService``.

100% SQLAlchemy: ningún path al filesystem. La migración del
``jobs.json`` legacy es responsabilidad de
``scripts/migrate_json_to_db.py`` (T19.7) — fuera del módulo.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from ossflow_service_kit.db import init_db, session_scope

from ..db import LegacyJobRow
from ..models import JobStatus, LegacyJob
from .._internal.orphan_recovery import mark_running_as_failed

log = logging.getLogger(__name__)


def _row_to_dataclass(row: LegacyJobRow) -> LegacyJob:
    """Mapea fila SQL → ``LegacyJob`` dataclass."""
    return LegacyJob(
        job_id=row.job_id,
        job_type=row.job_type,
        video_path=row.video_path,
        status=row.status,
        progress=row.progress,
        message=row.message or "",
        created_at=row.created_at.isoformat() if row.created_at else datetime.now().isoformat(),
        completed_at=row.completed_at.isoformat() if row.completed_at else None,
        result=json.loads(row.result) if row.result else None,
    )


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


class LegacyJobsRepository:
    """Acceso CRUD a la tabla ``legacy_jobs``.

    No conoce asyncio, FastAPI ni filesystem. Solo SQL y serialización
    JSON del campo ``result``.
    """

    # --- arranque ---------------------------------------------------------

    def init_db_and_recover(self) -> int:
        """Crea tablas (idempotente) y marca jobs ``RUNNING``/``QUEUED`` como
        ``FAILED``. Devuelve el número de huérfanos recuperados.

        El import del JSON legacy NO ocurre aquí — vive en
        ``scripts/migrate_json_to_db.py`` (T19.7). Por eso no hay
        ``history_file`` en el constructor: este repositorio asume que
        cualquier dato viejo ya está en BD.
        """
        # Importar este módulo asegura que ``LegacyJobRow`` se registre en
        # ``Base.metadata`` antes de ``create_all()``.
        from .. import db  # noqa: F401

        try:
            init_db()
        except Exception as exc:  # noqa: BLE001
            log.warning("init_db failed: %s", exc)
            return 0
        return self._recover_orphans()

    def _recover_orphans(self) -> int:
        try:
            with session_scope() as s:
                rows = s.query(LegacyJobRow).filter(
                    LegacyJobRow.status.in_(
                        [JobStatus.RUNNING.value, JobStatus.QUEUED.value]
                    )
                ).all()

                def _setter(row: LegacyJobRow, status: str, _error: str, completed_at: str) -> None:
                    # ``LegacyJob`` no tiene campo ``error`` (a diferencia de
                    # ``BackgroundJob``); guardamos el motivo en ``message``
                    # para no perder la trazabilidad.
                    row.status = status
                    row.message = row.message or _error
                    if not row.completed_at:
                        row.completed_at = datetime.fromisoformat(completed_at)

                return mark_running_as_failed(rows, _setter)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to recover orphan legacy jobs: %s", exc)
            return 0

    # --- CRUD -------------------------------------------------------------

    def get(self, job_id: str) -> Optional[LegacyJob]:
        try:
            with session_scope() as s:
                row = s.get(LegacyJobRow, job_id)
                return _row_to_dataclass(row) if row else None
        except Exception as exc:  # noqa: BLE001
            log.warning("get(%s) failed: %s", job_id, exc)
            return None

    def list_all(self, type_filter: Optional[str] = None) -> list[LegacyJob]:
        try:
            with session_scope() as s:
                q = s.query(LegacyJobRow)
                if type_filter:
                    q = q.filter(LegacyJobRow.job_type == type_filter)
                q = q.order_by(LegacyJobRow.created_at.desc())
                return [_row_to_dataclass(row) for row in q.all()]
        except Exception as exc:  # noqa: BLE001
            log.warning("list_all failed: %s", exc)
            return []

    def upsert(self, job: LegacyJob) -> None:
        try:
            with session_scope() as s:
                created = _parse_dt(job.created_at) or datetime.now()
                completed = _parse_dt(job.completed_at)
                row = s.get(LegacyJobRow, job.job_id)
                if row is None:
                    s.add(LegacyJobRow(
                        job_id=job.job_id,
                        job_type=job.job_type,
                        video_path=job.video_path,
                        status=job.status,
                        progress=job.progress,
                        message=job.message,
                        result=json.dumps(job.result) if job.result else None,
                        created_at=created,
                        completed_at=completed,
                    ))
                else:
                    row.job_type = job.job_type
                    row.video_path = job.video_path
                    row.status = job.status
                    row.progress = job.progress
                    row.message = job.message
                    row.result = json.dumps(job.result) if job.result else None
                    row.completed_at = completed
        except Exception as exc:  # noqa: BLE001
            log.warning("upsert(%s) failed: %s", job.job_id, exc)
