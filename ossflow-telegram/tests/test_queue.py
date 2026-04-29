"""Tests for DownloadQueue / SyncQueue.

Uses fake DB + fake downloader. No aiosqlite/telethon required for the
queue-mechanics suite (FIFO, 1 worker, cancellation, partial cleanup).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pytest

from telegram_fetcher.models import DownloadJob
from telegram_fetcher.queue import DownloadQueue, JobEventBus, SyncQueue


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeDB:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.status_calls: list[tuple[str, str]] = []

    async def enqueue_download(self, job: DownloadJob) -> None:
        self.jobs[job.id] = job.model_dump(mode="json")
        self.jobs[job.id]["status"] = job.status

    async def mark_download_status(
        self, job_id: str, status: str, **kw: Any
    ) -> None:
        self.status_calls.append((job_id, status))
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status

    async def upsert_media(self, *_a, **_kw) -> None:
        return None


def _mk_job(job_id: str, n: int = 2) -> DownloadJob:
    now = datetime.now(timezone.utc)
    return DownloadJob(
        id=job_id,
        channel_id="C",
        author="John",
        title="Leglocks",
        message_ids=[100 + i for i in range(n)],
        status="queued",
        total=n,
        destination_dir="/tmp",
        created_at=now,
        updated_at=now,
    )


class TestFIFO:
    async def test_single_worker_processes_jobs_in_order(self) -> None:
        db = _FakeDB()
        bus = JobEventBus()
        order: list[str] = []
        done = asyncio.Event()

        async def handler(job: DownloadJob, q: DownloadQueue) -> None:
            order.append(job.id)
            if len(order) == 3:
                done.set()

        dq = DownloadQueue(db, bus, handler)  # type: ignore[arg-type]
        await dq.start()
        await dq.enqueue(_mk_job("j1"))
        await dq.enqueue(_mk_job("j2"))
        await dq.enqueue(_mk_job("j3"))
        await asyncio.wait_for(done.wait(), timeout=2.0)
        await dq.stop()
        assert order == ["j1", "j2", "j3"]

    async def test_only_one_runs_at_a_time(self) -> None:
        db = _FakeDB()
        bus = JobEventBus()
        active = {"n": 0, "max": 0}

        async def handler(job: DownloadJob, q: DownloadQueue) -> None:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
            await asyncio.sleep(0.05)
            active["n"] -= 1

        dq = DownloadQueue(db, bus, handler)  # type: ignore[arg-type]
        await dq.start()
        for i in range(3):
            await dq.enqueue(_mk_job(f"j{i}"))
        await asyncio.sleep(0.3)
        await dq.stop()
        assert active["max"] == 1


class TestCancelMidFile:
    async def test_cancel_in_progress_triggers_cancelled_error_and_cleans_partial(
        self, tmp_path: Path
    ) -> None:
        db = _FakeDB()
        bus = JobEventBus()
        partial = tmp_path / "partial.mp4"
        started = asyncio.Event()
        cancelled_seen = asyncio.Event()

        async def handler(job: DownloadJob, q: DownloadQueue) -> None:
            # Simulate the Downloader writing a partial file, then being
            # interrupted by CancelledError.
            partial.write_bytes(b"incomplete")
            started.set()
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                # Mimic Downloader partial cleanup.
                try:
                    partial.unlink()
                except OSError:
                    pass
                cancelled_seen.set()
                raise

        dq = DownloadQueue(db, bus, handler)  # type: ignore[arg-type]
        await dq.start()
        job = _mk_job("jX")
        await dq.enqueue(job)
        assert await asyncio.wait_for(started.wait(), timeout=1.0)
        assert partial.exists()
        await dq.cancel("jX")
        assert await asyncio.wait_for(cancelled_seen.wait(), timeout=1.0)
        await asyncio.sleep(0.05)
        await dq.stop()
        assert not partial.exists()
        assert any(s == "cancelled" for _, s in db.status_calls)

    async def test_cancel_queued_job_skips_it(self) -> None:
        db = _FakeDB()
        bus = JobEventBus()
        ran: list[str] = []
        start1 = asyncio.Event()
        release1 = asyncio.Event()

        async def handler(job: DownloadJob, q: DownloadQueue) -> None:
            ran.append(job.id)
            if job.id == "first":
                start1.set()
                await release1.wait()

        dq = DownloadQueue(db, bus, handler)  # type: ignore[arg-type]
        await dq.start()
        await dq.enqueue(_mk_job("first"))
        await dq.enqueue(_mk_job("second"))
        await start1.wait()
        # Cancel second while first still holds the worker.
        await dq.cancel("second")
        release1.set()
        await asyncio.sleep(0.1)
        await dq.stop()
        assert "first" in ran
        assert "second" not in ran
        assert ("second", "cancelled") in db.status_calls


class TestSyncQueue:
    async def test_sync_queue_fifo_single_worker(self) -> None:
        bus = JobEventBus()
        order: list[str] = []

        async def handler(job: dict, q: SyncQueue) -> None:
            order.append(job["id"])
            await asyncio.sleep(0.02)

        sq = SyncQueue(bus, handler)
        await sq.start()
        ids = [await sq.enqueue(f"ch{i}") for i in range(3)]
        for _ in range(100):
            if len(order) == 3:
                break
            await asyncio.sleep(0.02)
        await sq.stop()
        assert order == ids
