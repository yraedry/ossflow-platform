"""Tests for /api/pipeline/batch — server-side multi-season orchestration.

Why this exists: the previous "Procesar todo" loop ran in the browser. Closing
the tab between seasons stranded the queue. The batch endpoint moves
orchestration to the backend so it survives navigator close.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import pipeline as pmod
from api.pipeline import StepStatus, router


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(pmod, "HISTORY_FILE", tmp_path / "pipeline_history.json")
    p_snap = dict(pmod._pipelines)
    b_snap = dict(pmod._batches)
    bt_snap = dict(pmod._batch_tasks)
    bc_snap = dict(pmod._batch_cancel)
    pmod._pipelines.clear()
    pmod._batches.clear()
    pmod._batch_tasks.clear()
    pmod._batch_cancel.clear()
    yield
    pmod._pipelines.clear()
    pmod._batches.clear()
    pmod._batch_tasks.clear()
    pmod._batch_cancel.clear()
    pmod._pipelines.update(p_snap)
    pmod._batches.update(b_snap)
    pmod._batch_tasks.update(bt_snap)
    pmod._batch_cancel.update(bc_snap)


def test_batch_rejects_missing_paths(tmp_path):
    client = _make_app()
    res = client.post("/api/pipeline/batch", json={
        "name": "x", "paths": [], "steps": ["chapters"],
    })
    assert res.status_code == 400


def test_batch_rejects_invalid_steps(tmp_path):
    season = tmp_path / "Season_01"
    season.mkdir()
    client = _make_app()
    res = client.post("/api/pipeline/batch", json={
        "name": "x", "paths": [str(season)], "steps": ["bogus"],
    })
    assert res.status_code == 422
    assert "bogus" in res.json()["error"]


def test_batch_rejects_nonexistent_paths(tmp_path):
    client = _make_app()
    res = client.post("/api/pipeline/batch", json={
        "name": "x",
        "paths": [str(tmp_path / "does_not_exist")],
        "steps": ["chapters"],
    })
    assert res.status_code == 422
    assert "missing" in res.json()


def test_batch_creates_and_runs_seasons_sequentially(tmp_path):
    """Batch launches seasons one at a time, waiting for each to finish."""
    s1 = tmp_path / "Season_01"
    s2 = tmp_path / "Season_02"
    s1.mkdir()
    s2.mkdir()

    launch_order: list[str] = []

    async def fake_run_pipeline(pipeline, queue):
        launch_order.append(pipeline.path)
        # Simulate a quick successful run
        await asyncio.sleep(0.01)
        pipeline.status = StepStatus.COMPLETED
        for s in pipeline.steps:
            s.status = StepStatus.COMPLETED

    with patch.object(pmod, "_run_pipeline", fake_run_pipeline):
        client = _make_app()
        res = client.post("/api/pipeline/batch", json={
            "name": "test",
            "paths": [str(s1), str(s2)],
            "steps": ["chapters"],
            "options": {"mode": "oracle"},
        })
        assert res.status_code == 200
        batch_id = res.json()["batch_id"]

        # Poll until terminal
        for _ in range(50):
            r = client.get(f"/api/pipeline/batch/{batch_id}")
            data = r.json()
            if data["status"] in ("completed", "failed", "cancelled"):
                break
            import time
            time.sleep(0.05)

        assert data["status"] == "completed"
        assert data["total"] == 2
        assert len(data["pipeline_ids"]) == 2
        # Sequential: season 1 path should have launched before season 2
        assert launch_order == [str(s1), str(s2)]


def test_batch_continues_on_failure_when_flag_set(tmp_path):
    s1 = tmp_path / "Season_01"
    s2 = tmp_path / "Season_02"
    s1.mkdir()
    s2.mkdir()

    call_count = {"n": 0}

    async def fake_run_pipeline(pipeline, queue):
        call_count["n"] += 1
        await asyncio.sleep(0.01)
        # First season fails, second succeeds
        if call_count["n"] == 1:
            pipeline.status = StepStatus.FAILED
        else:
            pipeline.status = StepStatus.COMPLETED

    with patch.object(pmod, "_run_pipeline", fake_run_pipeline):
        client = _make_app()
        res = client.post("/api/pipeline/batch", json={
            "name": "test",
            "paths": [str(s1), str(s2)],
            "steps": ["chapters"],
            "continue_on_fail": True,
        })
        batch_id = res.json()["batch_id"]

        import time
        for _ in range(50):
            data = client.get(f"/api/pipeline/batch/{batch_id}").json()
            if data["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.05)

        # Batch finishes "completed" even though season 1 failed — the flag
        # says "keep going". last_error preserves the diagnostic.
        assert data["status"] == "completed"
        assert call_count["n"] == 2
        assert "season 1" in (data["last_error"] or "")


def test_batch_stops_on_failure_when_flag_unset(tmp_path):
    s1 = tmp_path / "Season_01"
    s2 = tmp_path / "Season_02"
    s1.mkdir()
    s2.mkdir()

    call_count = {"n": 0}

    async def fake_run_pipeline(pipeline, queue):
        call_count["n"] += 1
        await asyncio.sleep(0.01)
        pipeline.status = StepStatus.FAILED

    with patch.object(pmod, "_run_pipeline", fake_run_pipeline):
        client = _make_app()
        res = client.post("/api/pipeline/batch", json={
            "name": "test",
            "paths": [str(s1), str(s2)],
            "steps": ["chapters"],
            "continue_on_fail": False,
        })
        batch_id = res.json()["batch_id"]

        import time
        for _ in range(50):
            data = client.get(f"/api/pipeline/batch/{batch_id}").json()
            if data["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.05)

        assert data["status"] == "failed"
        assert call_count["n"] == 1  # Stopped after first failure


@pytest.mark.asyncio
async def test_batch_cancel_aborts_remaining_seasons(tmp_path):
    """Direct unit test of _run_batch: setting cancel mid-run stops the loop."""
    s1 = tmp_path / "Season_01"
    s2 = tmp_path / "Season_02"
    s3 = tmp_path / "Season_03"
    for s in (s1, s2, s3):
        s.mkdir()

    launched: list[str] = []

    async def fake_run_pipeline(pipeline, queue):
        launched.append(pipeline.path)
        # Quick success so the runner moves to the next iteration fast
        await asyncio.sleep(0)
        pipeline.status = StepStatus.COMPLETED

    batch = pmod.BatchInfo(
        batch_id="testbatch",
        name="t",
        paths=[str(s1), str(s2), str(s3)],
        steps=["chapters"],
        options={},
    )
    pmod._batches["testbatch"] = batch
    pmod._batch_cancel["testbatch"] = False

    with patch.object(pmod, "_run_pipeline", fake_run_pipeline):
        # Pre-flag cancel after first season by using a callback that flips
        # the cancel bit when launched is non-empty.
        original_launch = pmod._launch_pipeline_internal

        def cancelling_launch(path, steps_raw, options):
            result = original_launch(path, steps_raw, options)
            # Flip cancel right after the first season is launched
            if len(launched) >= 1:
                pmod._batch_cancel["testbatch"] = True
            return result

        with patch.object(pmod, "_launch_pipeline_internal", cancelling_launch):
            await pmod._run_batch(batch)

    assert batch.status == StepStatus.CANCELLED
    assert len(launched) < 3  # Did not launch all three


def test_batch_cancel_endpoint_409_when_terminal(tmp_path):
    """Cancel returns 409 once the batch is already done."""
    s1 = tmp_path / "Season_01"
    s1.mkdir()

    async def fake_run_pipeline(pipeline, queue):
        await asyncio.sleep(0)
        pipeline.status = StepStatus.COMPLETED

    with patch.object(pmod, "_run_pipeline", fake_run_pipeline):
        client = _make_app()
        res = client.post("/api/pipeline/batch", json={
            "name": "t", "paths": [str(s1)], "steps": ["chapters"],
        })
        batch_id = res.json()["batch_id"]

        import time
        for _ in range(50):
            data = client.get(f"/api/pipeline/batch/{batch_id}").json()
            if data["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.05)
        assert data["status"] == "completed"

        cr = client.post(f"/api/pipeline/batch/{batch_id}/cancel")
        assert cr.status_code == 409


def test_get_batch_404_for_unknown(tmp_path):
    client = _make_app()
    res = client.get("/api/pipeline/batch/nope")
    assert res.status_code == 404


def test_list_batches_returns_recent(tmp_path):
    s1 = tmp_path / "Season_01"
    s1.mkdir()

    async def fake_run_pipeline(pipeline, queue):
        await asyncio.sleep(0.01)
        pipeline.status = StepStatus.COMPLETED

    with patch.object(pmod, "_run_pipeline", fake_run_pipeline):
        client = _make_app()
        for _ in range(3):
            client.post("/api/pipeline/batch", json={
                "name": "x", "paths": [str(s1)], "steps": ["chapters"],
            })

        res = client.get("/api/pipeline/batch")
        assert res.status_code == 200
        assert len(res.json()["batches"]) == 3
