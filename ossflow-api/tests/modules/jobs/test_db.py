"""Tests del modelo SQLAlchemy ``LegacyJobRow``.

Verifica que ``init_db()`` crea la tabla cuando el módulo está importado y
que las columnas son las esperadas.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import inspect

from ossflow_service_kit.db import engine as eng_mod
from ossflow_service_kit.db import init_db, session_scope
from ossflow_service_kit.db import session as sess_mod

# Import explícito para que ``Base.metadata`` registre la tabla antes de
# ``init_db()``. En producción esto lo hace el repositorio.
from ossflow_api.modules.jobs.db import LegacyJobRow  # noqa: F401


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """BD fresca por test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    eng_mod.reset_engine()
    sess_mod.reset_factory()
    init_db()
    yield db_path
    eng_mod.reset_engine()
    sess_mod.reset_factory()


def test_init_db_creates_legacy_jobs_table(fresh_db):
    """Cuando el módulo está importado, ``init_db()`` crea la tabla."""
    engine = eng_mod.get_engine()
    insp = inspect(engine)
    assert "legacy_jobs" in insp.get_table_names()


def test_legacy_jobs_columns_match_dataclass_fields(fresh_db):
    """Las columnas de la tabla deben corresponder con los campos del
    dataclass ``LegacyJob`` (excepto la serialización de ``result`` a JSON).
    """
    engine = eng_mod.get_engine()
    insp = inspect(engine)
    cols = {c["name"] for c in insp.get_columns("legacy_jobs")}
    expected = {
        "job_id",
        "job_type",
        "video_path",
        "status",
        "progress",
        "message",
        "result",
        "created_at",
        "completed_at",
    }
    assert cols == expected


def test_legacy_jobs_has_status_created_index(fresh_db):
    engine = eng_mod.get_engine()
    insp = inspect(engine)
    indexes = {ix["name"] for ix in insp.get_indexes("legacy_jobs")}
    assert "idx_legacy_jobs_status_created" in indexes


def test_legacy_jobs_can_insert_and_query(fresh_db):
    """Roundtrip básico: insert, query, verificar valores."""
    with session_scope() as s:
        s.add(LegacyJobRow(
            job_id="abc123",
            job_type="dubbing",
            video_path="/media/x.mp4",
            status="queued",
            progress=0.0,
            message="",
            created_at=datetime.utcnow(),
        ))

    with session_scope() as s:
        row = s.get(LegacyJobRow, "abc123")
        assert row is not None
        assert row.job_type == "dubbing"
        assert row.video_path == "/media/x.mp4"
        assert row.status == "queued"
        assert row.completed_at is None


def test_legacy_jobs_primary_key_prevents_duplicate_id(fresh_db):
    """El job_id es PK; un segundo insert con el mismo id debe fallar."""
    from sqlalchemy.exc import IntegrityError

    with session_scope() as s:
        s.add(LegacyJobRow(
            job_id="dup",
            job_type="dubbing",
            video_path="/p.mp4",
            status="queued",
        ))

    with pytest.raises(IntegrityError):
        with session_scope() as s:
            s.add(LegacyJobRow(
                job_id="dup",
                job_type="dubbing",
                video_path="/q.mp4",
                status="queued",
            ))
