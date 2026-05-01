"""Tests del router ``/api/jobs/*`` (legacy con SSE)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.jobs._internal.scheduler import JobsScheduler
from ossflow_api.modules.jobs._internal.sse_hub import SseHub
from ossflow_api.modules.jobs.dependencies import (
    get_legacy_jobs_dispatch_table,
    get_legacy_jobs_service,
)
from ossflow_api.modules.jobs.models import JobStatus, LegacyJob
from ossflow_api.modules.jobs.repositories.legacy import LegacyJobsRepository
from ossflow_api.modules.jobs.routers.legacy import router as legacy_router
from ossflow_api.modules.jobs.services.legacy import LegacyJobsService


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    from ossflow_service_kit.db import engine as eng_mod, session as sess_mod
    eng_mod.reset_engine()
    sess_mod.reset_factory()

    repo = LegacyJobsRepository()
    svc = LegacyJobsService(repo, JobsScheduler(), SseHub())
    svc.init()

    # Dispatch table fake: cada runner solo marca COMPLETED.
    async def _runner_chapters(job: LegacyJob, body: dict) -> None:
        svc.update_status(job.job_id, JobStatus.COMPLETED, result={"ran": "chapters"})

    async def _runner_dubbing(job: LegacyJob, body: dict) -> None:
        svc.update_status(job.job_id, JobStatus.COMPLETED, result={"ran": "dubbing"})

    dispatch = {
        "chapters": _runner_chapters,
        "dubbing": _runner_dubbing,
    }

    app = FastAPI()
    app.include_router(legacy_router)
    app.dependency_overrides[get_legacy_jobs_service] = lambda: svc
    app.dependency_overrides[get_legacy_jobs_dispatch_table] = lambda: dispatch

    yield TestClient(app)

    eng_mod.reset_engine()
    sess_mod.reset_factory()


def _wait_for_status(client, job_id, target_status, timeout=2.0):
    """Helper: espera a que el job llegue a un status concreto."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        if r.status_code == 200 and r.json()["status"] == target_status:
            return r.json()
        time.sleep(0.02)
    return None


def test_create_job_with_unknown_type_returns_400(client):
    r = client.post("/api/jobs", json={"type": "unknown_type", "path": "/x.mp4"})
    assert r.status_code == 400
    assert "Unknown job type" in r.json()["detail"]


def test_create_job_returns_job_id_and_queued(client):
    r = client.post("/api/jobs", json={"type": "dubbing", "path": "/media/v.mp4"})
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["status"] in ("queued", "running", "completed")


def test_create_job_runs_to_completion(client):
    r = client.post("/api/jobs", json={"type": "dubbing", "path": "/x.mp4"})
    job_id = r.json()["job_id"]
    final = _wait_for_status(client, job_id, "completed")
    assert final is not None
    assert final["result"] == {"ran": "dubbing"}


def test_get_missing_job_returns_404(client):
    r = client.get("/api/jobs/ghost")
    assert r.status_code == 404


def test_list_jobs_returns_all_created(client):
    client.post("/api/jobs", json={"type": "chapters", "path": "/a.mp4"})
    client.post("/api/jobs", json={"type": "dubbing", "path": "/b.mp4"})
    r = client.get("/api/jobs")
    assert r.status_code == 200
    body = r.json()
    assert len(body["jobs"]) >= 2


def test_list_jobs_filters_by_type(client):
    client.post("/api/jobs", json={"type": "chapters", "path": "/a.mp4"})
    client.post("/api/jobs", json={"type": "dubbing", "path": "/b.mp4"})
    r = client.get("/api/jobs?type=dubbing")
    assert r.status_code == 200
    body = r.json()
    assert all(j["job_type"] == "dubbing" for j in body["jobs"])


def test_response_shape_matches_legacy_contract(client):
    """``GET /api/jobs/{id}`` debe devolver las mismas keys que el legacy."""
    r = client.post("/api/jobs", json={"type": "dubbing", "path": "/x.mp4"})
    job_id = r.json()["job_id"]
    final = _wait_for_status(client, job_id, "completed")
    expected = {
        "job_id", "job_type", "video_path", "status", "progress",
        "message", "created_at", "completed_at", "result",
    }
    assert set(final.keys()) == expected


def test_video_path_is_top_level_field_not_in_params(client):
    """Decisión de diseño: ``video_path`` tipado en raíz, NO anidado."""
    r = client.post("/api/jobs", json={"type": "dubbing", "path": "/media/important.mp4"})
    job_id = r.json()["job_id"]
    detail = _wait_for_status(client, job_id, "completed")
    assert detail["video_path"] == "/media/important.mp4"
    # No debe existir ``params`` (eso es BackgroundJob, no LegacyJob).
    assert "params" not in detail
