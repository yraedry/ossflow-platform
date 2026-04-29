"""Tests for /telegram/channels endpoints."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


pytestmark = pytest.mark.asyncio


def _async_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


class _FakeEntity:
    def __init__(self, id: int, username: str, title: str) -> None:
        self.id = id
        self.username = username
        self.title = title


def _wire_fake_telethon(fake_svc, *, entities: dict[str, _FakeEntity] | None = None,
                        missing: set[str] | None = None) -> MagicMock:
    """Attach a fake Telethon client to the FakeTelegramService.

    - ``entities``: handle -> FakeEntity (returned by get_entity)
    - ``missing``: handles that raise (simulating channel not found)
    """
    entities = entities or {}
    missing = missing or set()
    client = MagicMock()

    async def get_entity(handle: str) -> _FakeEntity:
        key = handle.lstrip("@")
        if key in missing:
            raise RuntimeError(f"no such channel: {handle}")
        if key in entities:
            return entities[key]
        raise RuntimeError(f"unknown entity {handle}")

    client.get_entity = AsyncMock(side_effect=get_entity)
    fake_svc.client = client
    fake_svc.ensure_ready = AsyncMock(return_value=client)
    return client


async def test_list_channels_empty_by_default(wired_app) -> None:
    app, *_ = wired_app
    async with _async_client(app) as client:
        r = await client.get("/telegram/channels")
    assert r.status_code == 200
    assert r.json() == []


async def test_add_channel_persists_and_returns_row(wired_app) -> None:
    app, db, fake_svc, _ = wired_app
    _wire_fake_telethon(
        fake_svc,
        entities={"bjjinstructionalsmma": _FakeEntity(111, "bjjinstructionalsmma", "BJJ MMA")},
    )

    async with _async_client(app) as client:
        r = await client.post("/telegram/channels", json={"username": "@bjjinstructionalsmma"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["channel_id"] == "111"
    assert body["username"] == "bjjinstructionalsmma"
    assert body["title"] == "BJJ MMA"

    # Persisted to DB
    rows = await db.list_channels()
    assert len(rows) == 1
    assert rows[0]["channel_id"] == "111"


async def test_add_channel_accepts_tme_url(wired_app) -> None:
    app, _db, fake_svc, _ = wired_app
    _wire_fake_telethon(
        fake_svc,
        entities={"foochan": _FakeEntity(222, "foochan", "Foo")},
    )
    async with _async_client(app) as client:
        r = await client.post(
            "/telegram/channels", json={"username": "https://t.me/foochan"}
        )
    assert r.status_code == 201
    assert r.json()["username"] == "foochan"


async def test_add_channel_duplicate_conflicts(wired_app) -> None:
    app, _db, fake_svc, _ = wired_app
    _wire_fake_telethon(
        fake_svc,
        entities={"dup": _FakeEntity(333, "dup", "Dup")},
    )
    async with _async_client(app) as client:
        r1 = await client.post("/telegram/channels", json={"username": "dup"})
        assert r1.status_code == 201
        r2 = await client.post("/telegram/channels", json={"username": "dup"})
    assert r2.status_code == 409


async def test_add_channel_not_found(wired_app) -> None:
    app, _db, fake_svc, _ = wired_app
    _wire_fake_telethon(fake_svc, missing={"ghost"})
    async with _async_client(app) as client:
        r = await client.post("/telegram/channels", json={"username": "ghost"})
    assert r.status_code == 404


async def test_add_channel_empty_username_rejected(wired_app) -> None:
    app, _db, fake_svc, _ = wired_app
    _wire_fake_telethon(fake_svc)
    async with _async_client(app) as client:
        r = await client.post("/telegram/channels", json={"username": "   "})
    assert r.status_code == 400


async def test_add_channel_requires_auth(wired_app) -> None:
    app, _db, fake_svc, _ = wired_app
    fake_svc.ensure_ready = AsyncMock(side_effect=RuntimeError("not authed"))
    async with _async_client(app) as client:
        r = await client.post("/telegram/channels", json={"username": "whatever"})
    assert r.status_code == 401


async def test_patch_channel_renames(wired_app) -> None:
    app, db, _svc, _ = wired_app
    await db.upsert_channel("500", "renamec", title="Old")
    async with _async_client(app) as client:
        r = await client.patch("/telegram/channels/500", json={"title": "New Title"})
    assert r.status_code == 200
    assert r.json()["title"] == "New Title"
    rows = await db.list_channels()
    assert rows[0]["title"] == "New Title"


async def test_patch_channel_not_found(wired_app) -> None:
    app, *_ = wired_app
    async with _async_client(app) as client:
        r = await client.patch("/telegram/channels/does-not-exist", json={"title": "x"})
    assert r.status_code == 404


async def test_delete_channel_cascades_media_and_thumbs(wired_app, tmp_path) -> None:
    app, db, _svc, _ = wired_app
    from datetime import datetime, timezone
    from telegram_fetcher.models import MediaItem

    await db.upsert_channel("777", "delchan", title="Del")

    # Create a fake thumbnail file and a media row pointing at it.
    thumb = tmp_path / "777_1.jpg"
    thumb.write_bytes(b"jpeg")
    item = MediaItem(
        channel_id="777",
        message_id=1,
        caption=None,
        filename="v.mp4",
        size_bytes=10,
        mime_type="video/mp4",
        date=datetime.now(timezone.utc),
        author="A",
        title="T",
        chapter_num=1,
    )
    await db.upsert_media(item)
    await db.set_thumbnail_path("777", 1, str(thumb))

    async with _async_client(app) as client:
        r = await client.delete("/telegram/channels/777")
    assert r.status_code == 204

    # Channel gone
    assert await db.list_channels() == []
    # Media gone
    assert await db.list_media(channel_id="777") == []
    # Thumbnail file removed
    assert not thumb.exists()


async def test_delete_channel_not_found(wired_app) -> None:
    app, *_ = wired_app
    async with _async_client(app) as client:
        r = await client.delete("/telegram/channels/nope")
    assert r.status_code == 404


async def test_sync_returns_job_id(wired_app) -> None:
    app, *_ = wired_app
    sq = app.state.sync_queue

    async def _noop(job, queue):
        await queue.publish(job["id"], {"type": "done", "data": {}})

    sq._handler = _noop  # type: ignore[attr-defined]

    async with _async_client(app) as client:
        r = await client.post(
            "/telegram/channels/bjjinstructionalsmma/sync", json={"limit": 10}
        )
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body and isinstance(body["job_id"], str)


async def test_sync_sse_stream_emits_progress_and_done(wired_app) -> None:
    app, *_ = wired_app
    sq = app.state.sync_queue

    async def _fake(job, queue):
        await asyncio.sleep(0.05)
        await queue.publish(job["id"], {"type": "progress", "data": {"scanned": 5}})
        await queue.publish(job["id"], {"type": "done", "data": {"scanned": 5, "new": 5}})

    sq._handler = _fake  # type: ignore[attr-defined]

    async with _async_client(app) as client:
        r = await client.post("/telegram/channels/bjjinstructionalsmma/sync", json={})
        job_id = r.json()["job_id"]

        url = f"/telegram/channels/bjjinstructionalsmma/sync/{job_id}/events"
        events_seen: list[str] = []

        async def _consume() -> None:
            async with client.stream("GET", url) as s:
                assert s.status_code == 200
                async for line in s.aiter_lines():
                    if line.startswith("event:"):
                        events_seen.append(line.split(":", 1)[1].strip())
                    if "done" in events_seen:
                        return

        await asyncio.wait_for(_consume(), timeout=5.0)

    assert "progress" in events_seen
    assert "done" in events_seen
