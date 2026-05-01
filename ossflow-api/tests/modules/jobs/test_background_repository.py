"""Tests de BackgroundJobsRepository."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from ossflow_service_kit.db import engine as eng_mod
from ossflow_service_kit.db import session as sess_mod

from ossflow_api.modules.jobs.models import BackgroundJob, JobStatus
from ossflow_api.modules.jobs.repositories.background import (
    BackgroundJobsRepository,
    MAX_ENTRIES,
)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Repositorio con BD fresca por test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    eng_mod.reset_engine()
    sess_mod.reset_factory()
    r = BackgroundJobsRepository()
    r.init_db_and_recover()
    yield r
    eng_mod.reset_engine()
    sess_mod.reset_factory()


# ---------------------------------------------------------------------------
# init_db_and_recover
# ---------------------------------------------------------------------------


def test_init_creates_tables_and_returns_zero_orphans(tmp_path, monkeypatch):
    db_path = tmp_path / "fresh.db"
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    eng_mod.reset_engine()
    sess_mod.reset_factory()

    r = BackgroundJobsRepository()
    recovered = r.init_db_and_recover()

    assert recovered == 0
    eng_mod.reset_engine()
    sess_mod.reset_factory()


def test_init_recovers_orphans_running_to_failed(repo):
    """Jobs RUNNING/QUEUED al arrancar deben pasar a FAILED."""
    repo.upsert(BackgroundJob(id="r1", type="cleanup_scan", status="running"))
    repo.upsert(BackgroundJob(id="r2", type="cleanup_scan", status="queued"))
    repo.upsert(BackgroundJob(id="r3", type="cleanup_scan", status="completed"))

    # Forzamos un nuevo repo sobre la misma BD para simular reinicio.
    r2 = BackgroundJobsRepository()
    recovered = r2.init_db_and_recover()

    assert recovered == 2
    j1 = r2.get("r1")
    j2 = r2.get("r2")
    j3 = r2.get("r3")
    assert j1.status == "failed"
    assert j1.error == "interrupted: server restarted"
    assert j2.status == "failed"
    assert j3.status == "completed"


def test_import_legacy_json_once_then_renames_to_bak(tmp_path, monkeypatch):
    """Legacy ``background_jobs.json`` se importa al primer init y se
    renombra a ``.bak``. La segunda invocación es no-op."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    eng_mod.reset_engine()
    sess_mod.reset_factory()

    history = tmp_path / "background_jobs.json"
    history.write_text(json.dumps([
        {"id": "leg1", "type": "cleanup_scan", "status": "completed",
         "progress": 100.0, "params": {"path": "/x"}, "result": {"ok": True},
         "created_at": datetime.now().isoformat()},
    ]), encoding="utf-8")

    r = BackgroundJobsRepository(history_file=history)
    r.init_db_and_recover()

    # Backup creado, original borrado.
    assert not history.exists()
    assert (tmp_path / "background_jobs.json.bak").exists()

    # Job en BD.
    job = r.get("leg1")
    assert job is not None
    assert job.status == "completed"
    assert job.params == {"path": "/x"}
    assert job.result == {"ok": True}

    eng_mod.reset_engine()
    sess_mod.reset_factory()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_get_returns_none_for_missing(repo):
    assert repo.get("ghost") is None


def test_upsert_then_get_roundtrip(repo):
    repo.upsert(BackgroundJob(
        id="abc",
        type="cleanup_scan",
        status="running",
        progress=42.0,
        message="scanning",
        params={"path": "/media"},
    ))
    job = repo.get("abc")
    assert job is not None
    assert job.id == "abc"
    assert job.type == "cleanup_scan"
    assert job.status == "running"
    assert job.progress == 42.0
    assert job.message == "scanning"
    assert job.params == {"path": "/media"}


def test_upsert_overwrites_existing(repo):
    repo.upsert(BackgroundJob(id="x", type="t1", status="queued"))
    repo.upsert(BackgroundJob(id="x", type="t1", status="completed", progress=100.0))
    job = repo.get("x")
    assert job.status == "completed"
    assert job.progress == 100.0


def test_list_all_orders_by_created_at_desc(repo):
    # Insertamos con created_at distintos para que el orden sea predecible.
    for i in range(3):
        repo.upsert(BackgroundJob(
            id=f"j{i}",
            type="cleanup_scan",
            status="queued",
            created_at=f"2026-04-30T10:0{i}:00",
        ))
    jobs = repo.list_all()
    assert [j.id for j in jobs] == ["j2", "j1", "j0"]


def test_list_all_filters_by_type(repo):
    repo.upsert(BackgroundJob(id="a", type="cleanup_scan", status="queued"))
    repo.upsert(BackgroundJob(id="b", type="duplicates_scan", status="queued"))

    cleanup = repo.list_all(type_filter="cleanup_scan")
    duplicates = repo.list_all(type_filter="duplicates_scan")

    assert {j.id for j in cleanup} == {"a"}
    assert {j.id for j in duplicates} == {"b"}


def test_trim_to_keeps_only_n_most_recent(repo):
    for i in range(5):
        repo.upsert(BackgroundJob(
            id=f"j{i}",
            type="cleanup_scan",
            status="queued",
            created_at=f"2026-04-30T10:0{i}:00",
        ))
    repo.trim_to(2)
    jobs = repo.list_all()
    # Las 2 más recientes son j4 y j3.
    assert {j.id for j in jobs} == {"j3", "j4"}
