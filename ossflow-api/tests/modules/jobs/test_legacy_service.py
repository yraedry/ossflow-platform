"""Tests de LegacyJobsService."""

from __future__ import annotations

import asyncio

import pytest

from ossflow_service_kit.db import engine as eng_mod
from ossflow_service_kit.db import session as sess_mod

from ossflow_api.modules.jobs._internal.scheduler import JobsScheduler
from ossflow_api.modules.jobs._internal.sse_hub import SseHub
from ossflow_api.modules.jobs.models import JobStatus, LegacyJob
from ossflow_api.modules.jobs.repositories.legacy import LegacyJobsRepository
from ossflow_api.modules.jobs.services.legacy import LegacyJobsService


@pytest.fixture
def svc(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    eng_mod.reset_engine()
    sess_mod.reset_factory()
    repo = LegacyJobsRepository()
    s = LegacyJobsService(repo, JobsScheduler(), SseHub())
    s.init()
    yield s
    eng_mod.reset_engine()
    sess_mod.reset_factory()


def test_register_job_creates_in_queued(svc):
    job = svc.register_job("dubbing", "/media/x.mp4")
    assert job.status == "queued"
    assert job.job_type == "dubbing"
    assert job.video_path == "/media/x.mp4"
    persisted = svc.get(job.job_id)
    assert persisted is not None
    assert persisted.status == "queued"


def test_update_status_persists_completion(svc):
    job = svc.register_job("dubbing", "/x.mp4")
    svc.update_status(job.job_id, JobStatus.COMPLETED, result={"ok": True})
    final = svc.get(job.job_id)
    assert final.status == "completed"
    assert final.result == {"ok": True}
    assert final.progress == 100.0
    assert final.completed_at is not None


def test_update_status_failed_persists_message(svc):
    job = svc.register_job("dubbing", "/x.mp4")
    svc.update_status(job.job_id, JobStatus.FAILED, message="boom")
    final = svc.get(job.job_id)
    assert final.status == "failed"
    assert final.message == "boom"
    assert final.completed_at is not None


@pytest.mark.asyncio
async def test_emit_publishes_to_sse_hub_and_persists_status(svc):
    job = svc.register_job("dubbing", "/x.mp4")

    await svc.emit(job.job_id, {"status": "running", "message": "starting"})

    persisted = svc.get(job.job_id)
    assert persisted.status == "running"
    assert persisted.message == "starting"

    # Suscriptor debe ver el evento.
    agen = svc.subscribe_events(job.job_id)
    evt = await asyncio.wait_for(agen.__anext__(), timeout=1.0)
    assert evt == {"status": "running", "message": "starting"}


@pytest.mark.asyncio
async def test_spawn_runner_executes_and_marks_completion(svc):
    job = svc.register_job("dubbing", "/x.mp4")
    completed = asyncio.Event()

    async def runner(j: LegacyJob) -> None:
        svc.update_status(j.job_id, JobStatus.COMPLETED, result={"final": True})
        completed.set()

    svc.spawn_runner(job, runner)
    await asyncio.wait_for(completed.wait(), timeout=1.0)
    final = svc.get(job.job_id)
    assert final.status == "completed"
    assert final.result == {"final": True}


@pytest.mark.asyncio
async def test_spawn_runner_handles_exceptions_as_failed(svc):
    job = svc.register_job("dubbing", "/x.mp4")

    async def runner(j: LegacyJob) -> None:
        raise ValueError("kaboom")

    svc.spawn_runner(job, runner)

    # Espera transición a FAILED.
    for _ in range(100):
        if (j := svc.get(job.job_id)) and j.status == "failed":
            break
        await asyncio.sleep(0.01)
    final = svc.get(job.job_id)
    assert final.status == "failed"
    assert "kaboom" in final.message


def test_list_all_filters_by_type(svc):
    svc.register_job("dubbing", "/a.mp4")
    svc.register_job("chapters", "/b.mp4")
    svc.register_job("dubbing", "/c.mp4")

    dubbing = svc.list_all(type_filter="dubbing")
    chapters = svc.list_all(type_filter="chapters")
    assert len(dubbing) == 2
    assert len(chapters) == 1
