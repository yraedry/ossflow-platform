"""SSE heartbeat tests: streams emit ``: keepalive`` when idle."""

import asyncio
import json

import pytest


@pytest.mark.asyncio
async def test_job_events_emits_keepalive(monkeypatch):
    """When the queue has no events, the stream must yield a ``: keepalive``
    comment instead of closing. We patch ``asyncio.wait_for`` to time out
    immediately so the test is fast and deterministic."""
    from api import app as app_mod

    real_wait_for = asyncio.wait_for

    async def fast_wait_for(aw, timeout):
        return await real_wait_for(aw, timeout=0.05)

    monkeypatch.setattr("api.app.asyncio.wait_for", fast_wait_for)

    from api.app import JobInfo, JobStatus, _jobs, _job_events, api_job_events

    jid = "hb-test-1"
    _jobs[jid] = JobInfo(
        job_id=jid, job_type="chapters", video_path="/tmp/x.mkv",
        status=JobStatus.RUNNING,
    )
    q = asyncio.Queue()
    _job_events[jid] = q

    response = await api_job_events(jid)
    agen = response.body_iterator

    # First yield should be a keepalive comment (no events pending).
    first = await agen.__anext__()
    if isinstance(first, bytes):
        first = first.decode()
    assert first.startswith(": keepalive"), f"expected keepalive, got {first!r}"

    # Push a completion event → stream closes.
    await q.put({"status": "completed", "type": "status"})
    # May get another keepalive before completion due to timing; drain up to 3
    drained = []
    for _ in range(3):
        try:
            chunk = await asyncio.wait_for(agen.__anext__(), timeout=1.0)
        except (StopAsyncIteration, asyncio.TimeoutError):
            break
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        drained.append(chunk)
        if "completed" in chunk:
            break
    assert any("completed" in c for c in drained), drained

    _jobs.pop(jid, None)
    _job_events.pop(jid, None)


def test_emit_infers_type_log(monkeypatch):
    from api.app import _infer_event_type
    assert _infer_event_type({"message": "hi"}) == "log"
    assert _infer_event_type({"status": "running"}) == "status"
    assert _infer_event_type({"progress": 42}) == "progress"
    assert _infer_event_type({"type": "log", "progress": 1}) == "log"
