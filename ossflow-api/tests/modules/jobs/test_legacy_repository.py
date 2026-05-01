"""Tests de LegacyJobsRepository."""

from __future__ import annotations

import pytest

from ossflow_service_kit.db import engine as eng_mod
from ossflow_service_kit.db import session as sess_mod

from ossflow_api.modules.jobs.models import LegacyJob
from ossflow_api.modules.jobs.repositories.legacy import LegacyJobsRepository


@pytest.fixture
def repo(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    eng_mod.reset_engine()
    sess_mod.reset_factory()
    r = LegacyJobsRepository()
    r.init_db_and_recover()
    yield r
    eng_mod.reset_engine()
    sess_mod.reset_factory()


def test_init_creates_tables_and_returns_zero_orphans(tmp_path, monkeypatch):
    db_path = tmp_path / "fresh.db"
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    eng_mod.reset_engine()
    sess_mod.reset_factory()

    r = LegacyJobsRepository()
    recovered = r.init_db_and_recover()

    assert recovered == 0
    eng_mod.reset_engine()
    sess_mod.reset_factory()


def test_recover_marks_orphans_as_failed(repo):
    """Jobs RUNNING/QUEUED al arrancar deben pasar a FAILED."""
    repo.upsert(LegacyJob(job_id="r1", job_type="dubbing", video_path="/a.mp4", status="running"))
    repo.upsert(LegacyJob(job_id="r2", job_type="dubbing", video_path="/b.mp4", status="queued"))
    repo.upsert(LegacyJob(job_id="r3", job_type="dubbing", video_path="/c.mp4", status="completed"))

    r2 = LegacyJobsRepository()
    recovered = r2.init_db_and_recover()
    assert recovered == 2

    j1 = r2.get("r1")
    j2 = r2.get("r2")
    j3 = r2.get("r3")
    assert j1.status == "failed"
    assert j2.status == "failed"
    assert j3.status == "completed"


def test_get_returns_none_for_missing(repo):
    assert repo.get("ghost") is None


def test_upsert_and_get_roundtrip(repo):
    repo.upsert(LegacyJob(
        job_id="abc",
        job_type="dubbing",
        video_path="/media/x.mp4",
        status="running",
        progress=42.0,
        message="halfway",
    ))
    job = repo.get("abc")
    assert job is not None
    assert job.job_id == "abc"
    assert job.job_type == "dubbing"
    assert job.video_path == "/media/x.mp4"
    assert job.status == "running"
    assert job.progress == 42.0
    assert job.message == "halfway"


def test_upsert_overwrites_existing(repo):
    repo.upsert(LegacyJob(job_id="x", job_type="dubbing", video_path="/p.mp4", status="queued"))
    repo.upsert(LegacyJob(
        job_id="x", job_type="dubbing", video_path="/p.mp4",
        status="completed", progress=100.0, result={"ok": True},
    ))
    job = repo.get("x")
    assert job.status == "completed"
    assert job.progress == 100.0
    assert job.result == {"ok": True}


def test_list_all_orders_by_created_at_desc(repo):
    for i in range(3):
        repo.upsert(LegacyJob(
            job_id=f"j{i}",
            job_type="dubbing",
            video_path=f"/v{i}.mp4",
            status="queued",
            created_at=f"2026-04-30T10:0{i}:00",
        ))
    jobs = repo.list_all()
    assert [j.job_id for j in jobs] == ["j2", "j1", "j0"]


def test_list_all_filters_by_type(repo):
    repo.upsert(LegacyJob(job_id="a", job_type="dubbing", video_path="/a.mp4", status="queued"))
    repo.upsert(LegacyJob(job_id="b", job_type="chapters", video_path="/b.mp4", status="queued"))

    dubbing = repo.list_all(type_filter="dubbing")
    chapters = repo.list_all(type_filter="chapters")
    assert {j.job_id for j in dubbing} == {"a"}
    assert {j.job_id for j in chapters} == {"b"}
