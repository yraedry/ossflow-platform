"""Tests for GET /api/pipeline payload reduction, pagination + non-blocking
_save_history (see docs/reports/2026-04-13-performance-diagnosis.md §2)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import pipeline as pmod
from api.pipeline import PipelineInfo, StepInfo, StepStatus, router


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_pipelines(tmp_path, monkeypatch):
    # Isolate history file so the debounced writer cannot clobber real data.
    monkeypatch.setattr(pmod, "HISTORY_FILE", tmp_path / "pipeline_history.json")
    snapshot = dict(pmod._pipelines)
    pmod._pipelines.clear()
    yield
    pmod._pipelines.clear()
    pmod._pipelines.update(snapshot)


def _mk_pipeline(pid: str, status: StepStatus, created: datetime) -> PipelineInfo:
    steps = [
        StepInfo(
            name="chapters",
            status=StepStatus.COMPLETED,
            progress=100.0,
            message="done " + pid,
            started_at=created.isoformat(),
            completed_at=(created + timedelta(seconds=10)).isoformat(),
            diff={"added": ["a.mkv"], "removed": [], "modified": [], "truncated": False},
        ),
        StepInfo(name="subtitles", status=StepStatus.PENDING, progress=0.0),
    ]
    p = PipelineInfo(
        pipeline_id=pid,
        path=f"/tmp/{pid}.mkv",
        steps=steps,
        status=status,
        created_at=created.isoformat(),
        completed_at=(created + timedelta(seconds=30)).isoformat()
            if status == StepStatus.COMPLETED else None,
    )
    pmod._pipelines[pid] = p
    return p


def test_list_payload_is_summary_only_with_200_entries():
    base = datetime(2026, 4, 13, 12, 0, 0)
    for i in range(200):
        _mk_pipeline(f"p{i:03d}", StepStatus.COMPLETED, base + timedelta(minutes=i))

    client = _make_app()
    r = client.get("/api/pipeline")
    assert r.status_code == 200
    assert r.headers.get("X-Total-Count") == "200"
    data = r.json()
    assert len(data["pipelines"]) == 50  # default limit
    first = data["pipelines"][0]
    # Summary fields present
    assert set(first.keys()) == {
        "pipeline_id", "path", "status", "created_at", "completed_at", "steps"
    }
    # Steps are summary (name/status/progress only) — NO diff, message, started_at
    for step in first["steps"]:
        assert set(step.keys()) == {"name", "status", "progress"}
    # Sorted desc by created_at — p199 is the newest.
    assert first["pipeline_id"] == "p199"


def test_list_respects_limit_and_offset():
    base = datetime(2026, 4, 13, 12, 0, 0)
    for i in range(10):
        _mk_pipeline(f"p{i}", StepStatus.COMPLETED, base + timedelta(minutes=i))

    client = _make_app()
    r = client.get("/api/pipeline?limit=3&offset=2")
    assert r.status_code == 200
    assert r.headers["X-Total-Count"] == "10"
    data = r.json()
    assert len(data["pipelines"]) == 3
    # Desc: p9,p8,p7,p6,p5... offset 2 => p7,p6,p5
    assert [p["pipeline_id"] for p in data["pipelines"]] == ["p7", "p6", "p5"]


def test_list_filters_by_status():
    base = datetime(2026, 4, 13, 12, 0, 0)
    _mk_pipeline("running1", StepStatus.RUNNING, base)
    _mk_pipeline("done1", StepStatus.COMPLETED, base + timedelta(minutes=1))
    _mk_pipeline("running2", StepStatus.RUNNING, base + timedelta(minutes=2))
    _mk_pipeline("failed1", StepStatus.FAILED, base + timedelta(minutes=3))

    client = _make_app()
    r = client.get("/api/pipeline?status=running")
    assert r.status_code == 200
    assert r.headers["X-Total-Count"] == "2"
    ids = {p["pipeline_id"] for p in r.json()["pipelines"]}
    assert ids == {"running1", "running2"}


def test_detail_endpoint_still_returns_full_payload():
    base = datetime(2026, 4, 13, 12, 0, 0)
    _mk_pipeline("detail1", StepStatus.COMPLETED, base)
    client = _make_app()
    r = client.get("/api/pipeline/detail1")
    assert r.status_code == 200
    data = r.json()
    # Detail preserves full step info (diff, message, started_at)
    first_step = data["steps"][0]
    assert "diff" in first_step
    assert "message" in first_step
    assert "started_at" in first_step


def test_save_history_is_non_blocking():
    """_save_history must return essentially immediately — disk I/O runs
    on a daemon thread; bursts are debounced."""
    base = datetime(2026, 4, 13, 12, 0, 0)
    # Populate a non-trivial state (200 pipelines, each with 2 steps + diff)
    for i in range(200):
        _mk_pipeline(f"p{i:03d}", StepStatus.COMPLETED, base + timedelta(minutes=i))

    # Reset debounce state so the first call hits the immediate-write path.
    pmod._save_last_write = 0.0
    if pmod._save_timer is not None:
        pmod._save_timer.cancel()
        pmod._save_timer = None

    t0 = time.perf_counter()
    for _ in range(20):
        pmod._save_history()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 20 invocations must complete in well under 100ms — they spawn a thread
    # once then schedule a trailing timer; no synchronous disk writes on the
    # caller thread.
    assert elapsed_ms < 100, f"_save_history blocked caller for {elapsed_ms:.1f}ms"
