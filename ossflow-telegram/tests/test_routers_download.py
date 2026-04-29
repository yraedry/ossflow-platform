"""Tests for /telegram/download endpoints."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from telegram_fetcher.models import MediaItem


pytestmark = pytest.mark.asyncio


def _async_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _seed_two_chapters(db) -> None:
    for mid, ch in [(10, 1), (11, 2)]:
        await db.upsert_media(MediaItem(
            channel_id="C1",
            message_id=mid,
            caption=f"ch{ch}",
            filename=f"f{mid}.mp4",
            size_bytes=1000,
            mime_type="video/mp4",
            date=datetime(2026, 4, 1, 12, mid, tzinfo=timezone.utc),
            author="Danaher",
            title="Leglocks",
            chapter_num=ch,
        ))


async def test_enqueue_download_returns_job_id(wired_app) -> None:
    app, db, *_ = wired_app
    await _seed_two_chapters(db)

    # Replace handler so the worker doesn't touch telegram.
    done_evt = asyncio.Event()

    async def _noop(job, queue):
        await queue.publish(job.id, {"type": "done", "data": {"job_id": job.id}})
        await db.mark_download_status(job.id, "done")
        done_evt.set()

    app.state.download_queue._handler = _noop  # type: ignore[attr-defined]

    async with _async_client(app) as client:
        r = await client.post(
            "/telegram/download",
            json={"channel_id": "C1", "author": "Danaher", "title": "Leglocks"},
        )
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    await asyncio.wait_for(done_evt.wait(), timeout=3.0)


async def test_enqueue_download_404_when_no_media(wired_app) -> None:
    app, *_ = wired_app
    async with _async_client(app) as client:
        r = await client.post(
            "/telegram/download",
            json={"channel_id": "X", "author": "nobody", "title": "void"},
        )
    assert r.status_code == 404


async def test_download_sse_events(wired_app) -> None:
    app, db, *_ = wired_app
    await _seed_two_chapters(db)

    async def _fake(job, queue):
        await asyncio.sleep(0.05)
        await queue.publish(job.id, {"type": "progress", "data": {"overall_pct": 50.0}})
        await queue.publish(job.id, {"type": "done", "data": {"job_id": job.id}})
        await db.mark_download_status(job.id, "done")

    app.state.download_queue._handler = _fake  # type: ignore[attr-defined]

    async with _async_client(app) as client:
        r = await client.post(
            "/telegram/download",
            json={"channel_id": "C1", "author": "Danaher", "title": "Leglocks"},
        )
        job_id = r.json()["job_id"]

        events_seen: list[str] = []

        async def _consume() -> None:
            async with client.stream("GET", f"/telegram/download/{job_id}/events") as s:
                assert s.status_code == 200
                async for line in s.aiter_lines():
                    if line.startswith("event:"):
                        events_seen.append(line.split(":", 1)[1].strip())
                    if "done" in events_seen:
                        return

        await asyncio.wait_for(_consume(), timeout=5.0)

    assert "progress" in events_seen
    assert "done" in events_seen


async def test_cancel_happy(wired_app) -> None:
    app, db, *_ = wired_app
    await _seed_two_chapters(db)

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _slow(job, queue):
        try:
            started.set()
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    app.state.download_queue._handler = _slow  # type: ignore[attr-defined]

    async with _async_client(app) as client:
        r = await client.post(
            "/telegram/download",
            json={"channel_id": "C1", "author": "Danaher", "title": "Leglocks"},
        )
        job_id = r.json()["job_id"]
        await asyncio.wait_for(started.wait(), timeout=2.0)
        rc = await client.post(f"/telegram/download/{job_id}/cancel")
    assert rc.status_code == 200
    assert rc.json() == {"ok": True}
    await asyncio.wait_for(cancelled.wait(), timeout=2.0)


async def test_cancel_queued_job(wired_app) -> None:
    app, db, *_ = wired_app
    await _seed_two_chapters(db)

    # Make worker busy so second job stays queued.
    block = asyncio.Event()

    async def _blocker(job, queue):
        await block.wait()
        await db.mark_download_status(job.id, "done")

    app.state.download_queue._handler = _blocker  # type: ignore[attr-defined]

    async with _async_client(app) as client:
        r1 = await client.post(
            "/telegram/download",
            json={"channel_id": "C1", "author": "Danaher", "title": "Leglocks"},
        )
        r2 = await client.post(
            "/telegram/download",
            json={"channel_id": "C1", "author": "Danaher", "title": "Leglocks"},
        )
        j2 = r2.json()["job_id"]
        rc = await client.post(f"/telegram/download/{j2}/cancel")
    assert rc.status_code == 200
    block.set()


async def test_library_scan_triggered_on_done(wired_app) -> None:
    """When a job emits `done`, processor-api /api/library/scan is POSTed."""
    app, db, *_ = wired_app
    await _seed_two_chapters(db)

    # Wire completion callback (mirrors app.py startup).
    from telegram_fetcher.routers import download as download_mod
    bus = app.state.event_bus

    posted: list[str] = []

    async def _fake_scan(processor_api_url: str) -> None:
        posted.append(f"{processor_api_url.rstrip('/')}/api/library/scan")

    original_scan = download_mod.trigger_library_scan
    download_mod.trigger_library_scan = _fake_scan  # type: ignore[assignment]

    subscribed_evt = asyncio.Event()

    async def _cb(job_id: str) -> None:
        q = await bus.subscribe(job_id)
        subscribed_evt.set()
        try:
            while True:
                evt = await asyncio.wait_for(q.get(), timeout=5.0)
                if evt.get("type") == "__end__":
                    return
                if evt.get("type") == "done":
                    await download_mod.trigger_library_scan(app.state.config.processor_api_url)
                    return
        finally:
            await bus.unsubscribe(job_id, q)

    app.state.download_completion_cb = _cb

    async def _fast(job, queue):
        # Wait for the completion subscriber to attach before publishing.
        await asyncio.wait_for(subscribed_evt.wait(), timeout=3.0)
        await queue.publish(job.id, {"type": "done", "data": {"job_id": job.id}})
        await db.mark_download_status(job.id, "done")

    app.state.download_queue._handler = _fast  # type: ignore[attr-defined]

    try:
        async with _async_client(app) as client:
            r = await client.post(
                "/telegram/download",
                json={"channel_id": "C1", "author": "Danaher", "title": "Leglocks"},
            )
            assert r.status_code == 200
            # Wait for completion callback to fire.
            for _ in range(200):
                if posted:
                    break
                await asyncio.sleep(0.05)
    finally:
        download_mod.trigger_library_scan = original_scan

    assert any("/api/library/scan" in u for u in posted)


async def test_trigger_library_scan_posts_to_processor(monkeypatch) -> None:
    """trigger_library_scan hits `{URL}/api/library/scan` via httpx."""
    from telegram_fetcher.routers import download as download_mod

    captured: list[str] = []

    class _FakeAsyncClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw):
            captured.append(url)
            class _R:
                status_code = 200
            return _R()

    monkeypatch.setattr(download_mod.httpx, "AsyncClient", _FakeAsyncClient)
    await download_mod.trigger_library_scan("http://fake-processor:9999")
    assert captured == ["http://fake-processor:9999/api/library/scan"]


async def test_list_jobs_filter_by_status(wired_app) -> None:
    app, db, *_ = wired_app
    await _seed_two_chapters(db)

    async def _noop(job, queue):
        await db.mark_download_status(job.id, "running")  # non-standard but queryable

    app.state.download_queue._handler = _noop  # type: ignore[attr-defined]

    async with _async_client(app) as client:
        # Create 2 jobs then manually set their statuses
        r1 = await client.post(
            "/telegram/download",
            json={"channel_id": "C1", "author": "Danaher", "title": "Leglocks"},
        )
        r2 = await client.post(
            "/telegram/download",
            json={"channel_id": "C1", "author": "Danaher", "title": "Leglocks"},
        )
        await asyncio.sleep(0.1)
        await db.mark_download_status(r1.json()["job_id"], "running")
        await db.mark_download_status(r2.json()["job_id"], "done")

        r = await client.get("/telegram/download/jobs", params={"status": "running"})
    assert r.status_code == 200
    data = r.json()
    assert all(row["status"] == "running" for row in data)
    assert len(data) >= 1
