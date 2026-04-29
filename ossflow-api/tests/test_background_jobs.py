"""Tests for api.background_jobs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import background_jobs as bg
from api.background_jobs import (
    BackgroundJob,
    COMPLETED,
    FAILED,
    JobRegistry,
    QUEUED,
    RUNNING,
    router as bg_router,
)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Fresh registry with isolated DB + tmp legacy history file."""
    monkeypatch.setenv("BJJ_DB_PATH", str(tmp_path / "bjj.db"))
    from bjj_service_kit.db import engine as _eng, session as _sess
    _eng.reset_engine()
    _sess.reset_factory()

    history = tmp_path / "background_jobs.json"
    reg = JobRegistry(history_file=history)
    monkeypatch.setattr(bg, "registry", reg)
    return reg


@pytest.fixture
def client(registry):
    app = FastAPI()
    app.include_router(bg_router)
    return TestClient(app)


async def _wait_until(pred, timeout=2.0, step=0.01):
    elapsed = 0.0
    while elapsed < timeout:
        if pred():
            return True
        await asyncio.sleep(step)
        elapsed += step
    return False


def test_submit_transitions_queued_running_completed(registry):
    import threading

    async def scenario():
        gate = threading.Event()

        async def coro(update_progress):
            update_progress(10.0, "starting")
            # Poll the cross-thread gate without blocking the loop
            while not gate.is_set():
                await asyncio.sleep(0.01)
            update_progress(90.0, "finishing")
            return {"ok": True, "count": 3}

        job = registry.submit("cleanup_scan", coro, {"path": "/x"})
        assert job.status in (QUEUED, RUNNING)
        # Let runner start
        await _wait_until(lambda: job.status == RUNNING)
        assert job.status == RUNNING
        gate.set()
        await _wait_until(lambda: job.status == COMPLETED)
        assert job.status == COMPLETED
        assert job.result == {"ok": True, "count": 3}
        assert job.error is None
        assert job.progress == 100.0
        assert job.completed_at is not None

    asyncio.run(scenario())


def test_failed_job_marks_failed_with_error(registry):
    async def scenario():
        async def coro(update_progress):
            raise ValueError("boom")

        job = registry.submit("duplicates_scan", coro, {})
        await _wait_until(lambda: job.status == FAILED)
        assert job.status == FAILED
        assert "boom" in (job.error or "")
        assert job.result is None
        assert job.completed_at is not None

    asyncio.run(scenario())


def test_list_all_with_type_filter(registry):
    async def scenario():
        async def noop(update_progress):
            return {"ok": True}

        registry.submit("cleanup_scan", noop, {})
        registry.submit("duplicates_scan", noop, {})
        registry.submit("cleanup_scan", noop, {})
        await _wait_until(lambda: all(j.status == COMPLETED for j in registry.list_all()))

        all_jobs = registry.list_all()
        assert len(all_jobs) == 3

        only_cleanup = registry.list_all(type_filter="cleanup_scan")
        assert len(only_cleanup) == 2
        assert all(j.type == "cleanup_scan" for j in only_cleanup)

    asyncio.run(scenario())


def test_persistence_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BJJ_DB_PATH", str(tmp_path / "bjj.db"))
    from bjj_service_kit.db import engine as _eng, session as _sess
    _eng.reset_engine()
    _sess.reset_factory()

    async def scenario():
        history = tmp_path / "bg.json"
        reg1 = JobRegistry(history_file=history)

        async def coro(update_progress):
            return {"done": True}

        job = reg1.submit("cleanup_scan", coro, {"path": "/y"})
        await _wait_until(lambda: job.status == COMPLETED)

        # New registry (same DB) loads persisted state
        reg2 = JobRegistry(history_file=history)
        loaded = reg2.get(job.id)
        assert loaded is not None
        assert loaded.status == COMPLETED
        assert loaded.result == {"done": True}

    asyncio.run(scenario())


def test_orphan_running_job_marked_failed_on_load(tmp_path, monkeypatch):
    monkeypatch.setenv("BJJ_DB_PATH", str(tmp_path / "bjj.db"))
    from bjj_service_kit.db import engine as _eng, session as _sess, init_db, session_scope
    from bjj_service_kit.db.models import BackgroundJob as BGRow
    _eng.reset_engine()
    _sess.reset_factory()
    init_db()

    from datetime import datetime
    with session_scope() as s:
        s.add(BGRow(
            id="abc123",
            type="cleanup_scan",
            status=RUNNING,
            payload=json.dumps({"progress": 42.0, "message": "mid-scan", "params": {"path": "/p"}}),
            created_at=datetime(2024, 1, 1),
        ))

    history = tmp_path / "bg.json"  # not used, DB is source of truth
    reg = JobRegistry(history_file=history)
    loaded = reg.get("abc123")
    assert loaded is not None
    assert loaded.status == FAILED
    assert "interrupted" in (loaded.error or "").lower()
    assert loaded.completed_at is not None


def test_router_get_and_404(client, registry):
    async def noop(update_progress):
        return {"ok": True}

    async def scenario():
        job = registry.submit("cleanup_scan", noop, {"path": "/q"})
        await _wait_until(lambda: job.status == COMPLETED)
        return job

    job = asyncio.run(scenario())

    r = client.get(f"/api/background-jobs/{job.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == job.id
    assert body["status"] == COMPLETED
    assert body["result"] == {"ok": True}

    r404 = client.get("/api/background-jobs/does-not-exist")
    assert r404.status_code == 404

    r_list = client.get("/api/background-jobs")
    assert r_list.status_code == 200
    assert any(j["id"] == job.id for j in r_list.json()["jobs"])

    r_filter = client.get("/api/background-jobs?type=cleanup_scan")
    assert r_filter.status_code == 200
    assert all(j["type"] == "cleanup_scan" for j in r_filter.json()["jobs"])


def test_progress_callback_updates_job(registry):
    async def scenario():
        seen = {}

        async def coro(update_progress):
            update_progress(25.0, "quarter")
            seen["mid"] = True
            update_progress(75.0, "three-quarters")
            return {"ok": True}

        job = registry.submit("cleanup_scan", coro, {})
        await _wait_until(lambda: job.status == COMPLETED)
        # Final progress is set to 100 on completion
        assert job.progress == 100.0
        assert job.message in ("quarter", "three-quarters")
        assert seen.get("mid") is True

    asyncio.run(scenario())
