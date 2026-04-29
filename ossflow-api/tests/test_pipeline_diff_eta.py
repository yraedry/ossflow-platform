"""Tests for pipeline diff snapshots and ETA endpoint."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import pipeline as pmod
from api.pipeline import (
    PipelineInfo,
    StepInfo,
    StepStatus,
    _compute_diff,
    _snapshot_dir,
    router,
)
from fastapi import FastAPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Snapshot / diff
# ---------------------------------------------------------------------------

def test_snapshot_collects_sizes(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"\x00\x01\x02")
    snap = _snapshot_dir(tmp_path)
    assert "a.txt" in snap
    assert "sub/b.bin" in snap
    assert snap["a.txt"][0] == 5


def test_snapshot_none_dir_returns_empty():
    assert _snapshot_dir(None) == {}


def test_compute_diff_detects_added_modified_removed():
    before = {"keep.txt": (10, 1.0), "mod.txt": (20, 1.0), "gone.txt": (5, 1.0)}
    after = {"keep.txt": (10, 1.0), "mod.txt": (30, 2.0), "new.txt": (1, 1.0)}
    diff = _compute_diff(before, after)
    assert diff["added"] == ["new.txt"]
    assert diff["removed"] == ["gone.txt"]
    assert diff["modified"] == ["mod.txt"]
    assert diff["truncated"] is False


def test_compute_diff_truncation():
    before: dict[str, tuple[int, float]] = {}
    after = {f"f{i}.txt": (i, 1.0) for i in range(250)}
    diff = _compute_diff(before, after, limit=200)
    assert len(diff["added"]) == 200
    assert diff["truncated"] is True


# ---------------------------------------------------------------------------
# ETA endpoint
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_pipelines():
    snapshot = dict(pmod._pipelines)
    pmod._pipelines.clear()
    yield
    pmod._pipelines.clear()
    pmod._pipelines.update(snapshot)


def _mk_completed_pipeline(pid: str, durations: dict[str, float]) -> PipelineInfo:
    now = datetime.now()
    steps = []
    for name, dur in durations.items():
        s = StepInfo(
            name=name,
            status=StepStatus.COMPLETED,
            progress=100.0,
            started_at=now.isoformat(),
            completed_at=(now + timedelta(seconds=dur)).isoformat(),
        )
        steps.append(s)
    p = PipelineInfo(
        pipeline_id=pid,
        path="/tmp/v.mkv",
        steps=steps,
        status=StepStatus.COMPLETED,
        completed_at=now.isoformat(),
    )
    pmod._pipelines[pid] = p
    return p


def test_eta_returns_null_with_few_samples():
    _mk_completed_pipeline("p1", {"chapters": 60.0})
    _mk_completed_pipeline("p2", {"chapters": 70.0})
    client = _make_app()
    r = client.get("/api/pipeline/eta?steps=chapters,subtitles")
    assert r.status_code == 200
    data = r.json()
    assert data["per_step"]["chapters"] is None  # only 2 samples < 3
    assert data["per_step"]["subtitles"] is None
    assert data["total_seconds"] is None


def test_eta_returns_median_with_enough_samples():
    for i, d in enumerate([60.0, 120.0, 90.0, 100.0, 80.0]):
        _mk_completed_pipeline(f"p{i}", {"chapters": d})
    client = _make_app()
    r = client.get("/api/pipeline/eta?steps=chapters")
    assert r.status_code == 200
    data = r.json()
    # median of [60,120,90,100,80] == 90
    assert data["per_step"]["chapters"] == 90.0
    assert data["total_seconds"] == 90.0
    assert data["sample_counts"]["chapters"] == 5


def test_eta_rejects_invalid_steps():
    client = _make_app()
    r = client.get("/api/pipeline/eta?steps=bogus")
    assert r.status_code == 422


def test_eta_total_null_if_any_step_missing_samples():
    for i, d in enumerate([60.0, 120.0, 90.0]):
        _mk_completed_pipeline(f"p{i}", {"chapters": d, "subtitles": d * 2})
    # only 1 pipeline has dubbing
    _mk_completed_pipeline("pd", {"dubbing": 10.0})
    client = _make_app()
    r = client.get("/api/pipeline/eta?steps=chapters,subtitles,dubbing")
    data = r.json()
    assert data["per_step"]["chapters"] is not None
    assert data["per_step"]["subtitles"] is not None
    assert data["per_step"]["dubbing"] is None
    assert data["total_seconds"] is None


# ---------------------------------------------------------------------------
# Integration: _run_step emits step_diff and records diff on step
# ---------------------------------------------------------------------------

def test_run_step_emits_step_diff_and_records(monkeypatch, tmp_path: Path):
    # Set up a path with an initial file, and simulate the backend
    # "creating" a new file during the step execution via the fake stream.
    (tmp_path / "existing.txt").write_text("x")

    pipe = PipelineInfo(
        pipeline_id="tst",
        path=str(tmp_path / "v.mkv"),
        steps=[StepInfo(name="chapters")],
        options={"output_dir": str(tmp_path)},
    )

    class _FakeClient:
        base_url = "http://fake"

        async def run(self, payload):
            return "job-1"

        async def stream(self, remote_id):
            from api.event_normalizer import NormalizedEvent
            # Mid-stream, "create" a new output file
            (tmp_path / "new.txt").write_text("hello world")
            yield NormalizedEvent(kind="progress", status="running", progress=50.0, message="m", payload={})
            yield NormalizedEvent(kind="done", status="completed", progress=100.0, message="ok", payload={})

    def fake_cp(step_name, path, options, chained_path=None):
        return _FakeClient(), {}, False

    monkeypatch.setattr(pmod, "_client_and_payload", fake_cp)

    async def _run():
        # _emit now broadcasts to per-subscriber queues (fan-out) instead of
        # a single shared producer queue, so we subscribe before running.
        sub = pmod._subscribe(pipe.pipeline_id)
        try:
            queue: asyncio.Queue = asyncio.Queue()
            ok = await pmod._run_step(pipe, 0, queue)
            events = []
            while not sub.empty():
                events.append(sub.get_nowait())
            return ok, events
        finally:
            pmod._unsubscribe(pipe.pipeline_id, sub)

    ok, events = asyncio.run(_run())
    assert ok is True
    assert pipe.steps[0].diff is not None
    assert "new.txt" in pipe.steps[0].diff["added"]
    kinds = [e.get("type") for e in events]
    assert "step_diff" in kinds


def test_emit_fans_out_to_all_subscribers():
    """Multiple SSE subscribers on the same pipeline must each receive every
    event. Regression guard: a single shared queue caused later consumers
    (or StrictMode double-mounts) to steal live logs from the LogPanel.
    """
    pipe = PipelineInfo(
        pipeline_id="fanout",
        path="/tmp/x",
        steps=[StepInfo(name="chapters")],
    )

    async def _run():
        a = pmod._subscribe(pipe.pipeline_id)
        b = pmod._subscribe(pipe.pipeline_id)
        try:
            for i in range(5):
                await pmod._emit(
                    pipe, asyncio.Queue(),
                    {"type": "step_progress", "step": "chapters",
                     "message": f"line {i}", "progress": i * 20},
                )
            def drain(q):
                return [q.get_nowait() for _ in range(q.qsize())]
            return drain(a), drain(b)
        finally:
            pmod._unsubscribe(pipe.pipeline_id, a)
            pmod._unsubscribe(pipe.pipeline_id, b)

    ev_a, ev_b = asyncio.run(_run())
    assert len(ev_a) == 5 and len(ev_b) == 5
    assert [e["message"] for e in ev_a] == [f"line {i}" for i in range(5)]
    assert [e["message"] for e in ev_b] == [f"line {i}" for i in range(5)]
    seqs = [e["seq"] for e in ev_a]
    assert seqs == sorted(seqs) and len(set(seqs)) == 5
