"""Tests del SseHub."""

from __future__ import annotations

import asyncio

import pytest

from ossflow_api.modules.jobs._internal.sse_hub import SseHub


@pytest.mark.asyncio
async def test_register_creates_queue_idempotent():
    hub = SseHub()
    q1 = hub.register("job-1")
    q2 = hub.register("job-1")
    assert q1 is q2  # mismo objeto, idempotente


@pytest.mark.asyncio
async def test_publish_to_unregistered_id_is_noop():
    hub = SseHub()
    hub.publish("missing", {"foo": "bar"})  # no debe lanzar


@pytest.mark.asyncio
async def test_publish_then_subscribe_drains_in_order():
    hub = SseHub()
    hub.register("j")
    hub.publish("j", {"n": 1})
    hub.publish("j", {"n": 2})
    hub.publish("j", {"n": 3})

    received: list[dict] = []
    agen = hub.subscribe("j", keepalive_seconds=10.0)
    for _ in range(3):
        evt = await asyncio.wait_for(agen.__anext__(), timeout=1.0)
        assert evt is not None
        received.append(evt)
    assert received == [{"n": 1}, {"n": 2}, {"n": 3}]


@pytest.mark.asyncio
async def test_subscribe_emits_keepalive_on_timeout():
    """Sin eventos durante ``keepalive_seconds``, debe yield ``None``."""
    hub = SseHub()
    agen = hub.subscribe("j", keepalive_seconds=0.05)
    evt = await asyncio.wait_for(agen.__anext__(), timeout=1.0)
    assert evt is None


@pytest.mark.asyncio
async def test_close_removes_queue():
    hub = SseHub()
    hub.register("j")
    assert "j" in hub.known_ids()
    hub.close("j")
    assert "j" not in hub.known_ids()


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown():
    hub = SseHub()
    assert hub.get("ghost") is None
