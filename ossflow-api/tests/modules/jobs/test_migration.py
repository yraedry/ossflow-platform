"""Tests del script ``scripts.migrate_json_to_db`` para jobs.

Verifica:
* ``migrate_background_jobs`` importa entradas válidas y renombra a ``.bak``.
* ``migrate_legacy_jobs`` idem para ``jobs.json``.
* Idempotencia: dos ejecuciones consecutivas → la segunda no duplica.
* Robustez: archivo inexistente / contenido inválido → 0 imports sin
  romper.
"""

from __future__ import annotations

import json

import pytest

from ossflow_service_kit.db import engine as eng_mod
from ossflow_service_kit.db import init_db, session_scope
from ossflow_service_kit.db import session as sess_mod
from ossflow_service_kit.db.models import BackgroundJob as BackgroundJobRow

from ossflow_api.modules.jobs.db import LegacyJobRow

# Importar el módulo del script. Se hace desde aquí para asegurar que
# los modelos del kit y de modules.jobs están registrados antes.
from scripts import migrate_json_to_db as mig


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    eng_mod.reset_engine()
    sess_mod.reset_factory()
    init_db()
    yield tmp_path
    eng_mod.reset_engine()
    sess_mod.reset_factory()


# ---------------------------------------------------------------------------
# migrate_background_jobs
# ---------------------------------------------------------------------------


def test_background_jobs_missing_file_is_noop(fresh_db):
    imported = mig.migrate_background_jobs(fresh_db, dry_run=False)
    assert imported == 0


def test_background_jobs_imports_and_renames(fresh_db):
    src = fresh_db / "background_jobs.json"
    src.write_text(json.dumps([
        {"id": "a", "type": "cleanup_scan", "status": "completed",
         "progress": 100.0, "params": {"path": "/x"}, "result": {"ok": True}},
        {"id": "b", "type": "duplicates_scan", "status": "failed",
         "error": "boom", "params": {}},
    ]), encoding="utf-8")

    imported = mig.migrate_background_jobs(fresh_db, dry_run=False)
    assert imported == 2
    assert not src.exists()
    assert (fresh_db / "background_jobs.json.bak").exists()

    # Filas en BD.
    with session_scope() as s:
        rows = {r.id: r for r in s.query(BackgroundJobRow).all()}
        assert "a" in rows and "b" in rows
        assert rows["a"].status == "completed"
        assert rows["b"].error == "boom"


def test_background_jobs_dry_run_does_not_persist(fresh_db):
    src = fresh_db / "background_jobs.json"
    src.write_text(json.dumps([{"id": "a", "type": "x", "status": "queued"}]), encoding="utf-8")

    imported = mig.migrate_background_jobs(fresh_db, dry_run=True)
    assert imported == 1  # cuenta el "would import"
    # Pero no se persistió ni se renombró.
    assert src.exists()
    with session_scope() as s:
        rows = s.query(BackgroundJobRow).all()
        assert len(rows) == 0


def test_background_jobs_idempotent_second_run_imports_zero(fresh_db):
    src = fresh_db / "background_jobs.json"
    src.write_text(json.dumps([
        {"id": "a", "type": "cleanup_scan", "status": "completed", "params": {}},
    ]), encoding="utf-8")

    first = mig.migrate_background_jobs(fresh_db, dry_run=False)
    assert first == 1
    # Segunda ejecución: el archivo ya no existe.
    second = mig.migrate_background_jobs(fresh_db, dry_run=False)
    assert second == 0


def test_background_jobs_invalid_format_returns_zero(fresh_db):
    src = fresh_db / "background_jobs.json"
    src.write_text("{not json", encoding="utf-8")

    imported = mig.migrate_background_jobs(fresh_db, dry_run=False)
    assert imported == 0


# ---------------------------------------------------------------------------
# migrate_legacy_jobs
# ---------------------------------------------------------------------------


def test_legacy_jobs_missing_file_is_noop(fresh_db):
    imported = mig.migrate_legacy_jobs(fresh_db, dry_run=False)
    assert imported == 0


def test_legacy_jobs_imports_and_renames(fresh_db):
    src = fresh_db / "jobs.json"
    src.write_text(json.dumps({
        "j1": {
            "job_id": "j1",
            "job_type": "dubbing",
            "video_path": "/media/x.mp4",
            "status": "completed",
            "progress": 100.0,
            "result": {"output": "/media/x_DOBLADO.mkv"},
        },
        "j2": {
            "job_id": "j2",
            "job_type": "chapters",
            "video_path": "/media/y.mp4",
            "status": "failed",
            "message": "ffmpeg crashed",
        },
    }), encoding="utf-8")

    imported = mig.migrate_legacy_jobs(fresh_db, dry_run=False)
    assert imported == 2
    assert not src.exists()
    assert (fresh_db / "jobs.json.bak").exists()

    with session_scope() as s:
        rows = {r.job_id: r for r in s.query(LegacyJobRow).all()}
        assert "j1" in rows and "j2" in rows
        assert rows["j1"].video_path == "/media/x.mp4"
        assert rows["j2"].message == "ffmpeg crashed"


def test_legacy_jobs_skips_entries_without_video_path(fresh_db):
    src = fresh_db / "jobs.json"
    src.write_text(json.dumps({
        "ok": {"job_type": "dubbing", "video_path": "/x.mp4", "status": "queued"},
        "broken": {"job_type": "dubbing", "status": "queued"},  # sin video_path
    }), encoding="utf-8")

    imported = mig.migrate_legacy_jobs(fresh_db, dry_run=False)
    assert imported == 1  # solo "ok" — "broken" sin video_path se salta.

    with session_scope() as s:
        rows = {r.job_id for r in s.query(LegacyJobRow).all()}
        assert rows == {"ok"}


def test_legacy_jobs_idempotent_when_existing_in_db(fresh_db):
    """Si una entrada ya está en BD con el mismo job_id, no se duplica."""
    # Pre-pueblar BD.
    with session_scope() as s:
        s.add(LegacyJobRow(
            job_id="exists",
            job_type="dubbing",
            video_path="/old.mp4",
            status="completed",
        ))

    src = fresh_db / "jobs.json"
    src.write_text(json.dumps({
        "exists": {
            "job_type": "dubbing",
            "video_path": "/new.mp4",  # distinto del de BD
            "status": "completed",
        },
        "new": {
            "job_type": "dubbing",
            "video_path": "/y.mp4",
            "status": "queued",
        },
    }), encoding="utf-8")

    imported = mig.migrate_legacy_jobs(fresh_db, dry_run=False)
    assert imported == 1  # solo "new" — "exists" se preserva.

    with session_scope() as s:
        existing = s.get(LegacyJobRow, "exists")
        assert existing.video_path == "/old.mp4"  # preservado, no sobrescrito.


def test_legacy_jobs_invalid_format_returns_zero(fresh_db):
    src = fresh_db / "jobs.json"
    src.write_text("[]", encoding="utf-8")  # array, no objeto

    imported = mig.migrate_legacy_jobs(fresh_db, dry_run=False)
    assert imported == 0
