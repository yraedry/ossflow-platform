"""Tests for BackendClient (run + SSE stream)."""

from __future__ import annotations

import httpx
import pytest
import respx

from api.backend_client import BackendClient, BackendError, _parse_sse_block


@pytest.mark.asyncio
@respx.mock
async def test_run_returns_job_id():
    client = BackendClient("http://splitter.test")
    respx.post("http://splitter.test/run").mock(
        return_value=httpx.Response(200, json={"job_id": "abc123"})
    )
    jid = await client.run({"path": "/foo"})
    assert jid == "abc123"


@pytest.mark.asyncio
@respx.mock
async def test_run_raises_on_error():
    client = BackendClient("http://splitter.test")
    respx.post("http://splitter.test/run").mock(
        return_value=httpx.Response(500, text="boom")
    )
    with pytest.raises(BackendError):
        await client.run({})


@pytest.mark.asyncio
@respx.mock
async def test_stream_parses_sse_events():
    client = BackendClient("http://splitter.test")
    sse_body = (
        'data: {"status": "running", "progress": 0.1}\n\n'
        'data: {"status": "running", "progress": 0.5}\n\n'
        'data: {"status": "done", "progress": 1.0}\n\n'
    )
    respx.get("http://splitter.test/events/job1").mock(
        return_value=httpx.Response(
            200, text=sse_body, headers={"content-type": "text/event-stream"}
        )
    )
    events = []
    async for ev in client.stream("job1"):
        events.append(ev)
    assert len(events) == 3
    # stream now yields NormalizedEvent (kind attr must exist)
    assert all(hasattr(e, "kind") for e in events)
    assert events[0].kind == "progress"
    assert events[0].progress == 10.0  # 0.1 -> 10%
    assert events[-1].kind == "done"
    assert events[-1].status == "completed"


@pytest.mark.asyncio
@respx.mock
async def test_stream_accepts_bjj_service_kit_contract():
    """Events shaped as {"type","data"} must be normalized and terminate."""
    client = BackendClient("http://splitter.test")
    sse_body = (
        'data: {"type": "progress", "data": {"percent": 0.25, "message": "x"}}\n\n'
        'data: {"type": "error", "data": {"message": "boom"}}\n\n'
    )
    respx.get("http://splitter.test/events/j2").mock(
        return_value=httpx.Response(
            200, text=sse_body, headers={"content-type": "text/event-stream"}
        )
    )
    events = []
    async for ev in client.stream("j2"):
        events.append(ev)
    assert len(events) == 2
    assert events[0].kind == "progress"
    assert events[0].progress == 25.0
    assert events[1].kind == "error"
    assert events[1].message == "boom"


@pytest.mark.asyncio
@respx.mock
async def test_stream_404_on_reconnect_after_events_treated_as_clean_close():
    """Long-running jobs may finish + get reaped from the backend
    registry while the client is reconnecting after a read timeout.
    The reconnect lands on 404, but the stream had already yielded
    real events, so we treat it as a clean close instead of raising
    BackendError (which would surface as step_failed even though the
    job actually completed)."""
    client = BackendClient("http://dub.test")
    # First call: yields one event then closes without a terminal frame,
    # which triggers a reconnect attempt.
    sse_body = 'data: {"status": "running", "progress": 0.5}\n\n'
    route = respx.get("http://dub.test/events/long-job").mock(
        side_effect=[
            httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"}),
            httpx.Response(404, text="job_id not found"),
        ]
    )
    events = []
    async for ev in client.stream("long-job", max_reconnects=2):
        events.append(ev)
    assert len(events) == 1
    assert events[0].kind == "progress"
    assert route.call_count == 2  # original + reconnect


@pytest.mark.asyncio
@respx.mock
async def test_stream_404_on_first_connect_still_raises():
    """A 404 on the very first attempt (no events ever seen) is a real
    bug — the caller passed a bogus job_id. Keep raising so we don't
    mask programmer errors."""
    client = BackendClient("http://dub.test")
    respx.get("http://dub.test/events/nonexistent").mock(
        return_value=httpx.Response(404, text="job_id not found")
    )
    with pytest.raises(BackendError):
        async for _ in client.stream("nonexistent"):
            pass


def test_parse_sse_block_ignores_comments():
    assert _parse_sse_block([": heartbeat"]) is None


def test_parse_sse_block_parses_data():
    evt = _parse_sse_block(['data: {"a": 1}'])
    assert evt == {"a": 1}


def test_parse_sse_block_fallback_non_json():
    evt = _parse_sse_block(["data: hello world"])
    assert evt == {"raw": "hello world"}


def test_factory_reads_env(monkeypatch):
    from api import backend_client as bc

    bc.reset_clients()
    monkeypatch.setenv("SPLITTER_URL", "http://custom:9000")
    c = bc.splitter_client()
    assert c.base_url == "http://custom:9000"
