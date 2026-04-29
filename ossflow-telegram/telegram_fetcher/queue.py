"""Asyncio FIFO queues for download + sync jobs.

Single worker per queue. Jobs live in SQLite (``download_jobs`` table); the
in-memory queue only carries ``job_id`` strings. Cancellation mid-file is
supported by cancelling the asyncio Task running the job — ``Downloader``
honours :class:`asyncio.CancelledError` and deletes the partial file.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .browser import ChannelBrowser
from .db import Database
from .downloader import Downloader
from .errors import TelegramError
from .models import DownloadJob
from .naming import chapter_filename, instructional_dirname


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event bus (in-memory) for SSE subscribers.
# ---------------------------------------------------------------------------


class JobEventBus:
    """Per-job event fanout backed by asyncio.Queue.

    Multiple consumers can subscribe simultaneously (each gets its own queue).
    """

    def __init__(self) -> None:
        self._subs: Dict[str, List[asyncio.Queue[dict]]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, job_id: str, event: dict) -> None:
        async with self._lock:
            queues = list(self._subs.get(job_id, ()))
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # unbounded by default, defensive
                pass

    async def subscribe(self, job_id: str) -> "asyncio.Queue[dict]":
        q: asyncio.Queue[dict] = asyncio.Queue()
        async with self._lock:
            self._subs.setdefault(job_id, []).append(q)
        return q

    async def unsubscribe(self, job_id: str, q: "asyncio.Queue[dict]") -> None:
        async with self._lock:
            lst = self._subs.get(job_id)
            if not lst:
                return
            try:
                lst.remove(q)
            except ValueError:
                pass
            if not lst:
                self._subs.pop(job_id, None)

    async def close(self, job_id: str) -> None:
        """Notify subscribers that no more events will arrive."""
        await self.publish(job_id, {"type": "__end__", "data": {}})


# ---------------------------------------------------------------------------
# Download queue
# ---------------------------------------------------------------------------


DownloadHandler = Callable[[DownloadJob, "DownloadQueue"], Awaitable[None]]
SyncHandler = Callable[[dict, "SyncQueue"], Awaitable[None]]


class DownloadQueue:
    """FIFO asyncio queue with a single worker task.

    The worker calls a user-supplied coroutine (``handler``) for each job.
    ``handler`` receives the :class:`DownloadJob` and the queue (for event
    publishing). Exceptions propagate to the worker which marks the job
    ``failed``; :class:`asyncio.CancelledError` marks it ``cancelled``.
    """

    def __init__(
        self,
        db: Database,
        bus: JobEventBus,
        handler: DownloadHandler,
    ) -> None:
        self._db = db
        self._bus = bus
        self._handler = handler
        self._queue: asyncio.Queue[DownloadJob] = asyncio.Queue()
        self._worker: Optional[asyncio.Task[None]] = None
        self._current_job_id: Optional[str] = None
        self._current_task: Optional[asyncio.Task[None]] = None
        self._cancelled_ids: set[str] = set()

    # --- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker(), name="download-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._worker = None

    # --- enqueue / cancel ---------------------------------------------

    async def enqueue(self, job: DownloadJob) -> None:
        await self._db.enqueue_download(job)
        await self._queue.put(job)
        await self._bus.publish(job.id, {"type": "queued", "data": {"job_id": job.id}})

    async def cancel(self, job_id: str) -> bool:
        """Cancel a job. Returns True if action was taken.

        - Queued or not yet started: marked ``cancelled`` in DB; worker skips.
        - In progress: cancels the running asyncio.Task → ``CancelledError``
          inside the handler → partial file cleaned by :class:`Downloader`.
        """
        self._cancelled_ids.add(job_id)
        if self._current_job_id == job_id and self._current_task is not None:
            self._current_task.cancel()
            return True
        # Not the active one — mark in DB now so it's skipped on dequeue.
        try:
            await self._db.mark_download_status(job_id, "cancelled", error="cancelled by user")
        except Exception:  # noqa: BLE001
            log.exception("mark cancelled failed for %s", job_id)
        await self._bus.publish(job_id, {"type": "cancelled", "data": {"job_id": job_id}})
        await self._bus.close(job_id)
        return True

    # --- worker loop ---------------------------------------------------

    async def _run_worker(self) -> None:
        while True:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                return
            if job.id in self._cancelled_ids:
                self._cancelled_ids.discard(job.id)
                continue
            try:
                await self._process(job)
            except Exception:  # noqa: BLE001
                log.exception("worker iteration crashed")

    async def _process(self, job: DownloadJob) -> None:
        job_id = job.id
        self._current_job_id = job_id
        handler_coro = self._handler(job, self)
        self._current_task = asyncio.create_task(handler_coro, name=f"dl-{job_id}")
        try:
            await self._current_task
        except asyncio.CancelledError:
            await self._db.mark_download_status(job_id, "cancelled", error="cancelled by user")
            await self._bus.publish(job_id, {"type": "cancelled", "data": {"job_id": job_id}})
        except Exception as exc:  # noqa: BLE001
            log.exception("job %s failed", job_id)
            await self._db.mark_download_status(job_id, "failed", error=str(exc))
            await self._bus.publish(
                job_id, {"type": "error", "data": {"message": str(exc)}}
            )
        finally:
            self._current_job_id = None
            self._current_task = None
            self._cancelled_ids.discard(job_id)
            await self._bus.close(job_id)

    async def publish(self, job_id: str, event: dict) -> None:
        await self._bus.publish(job_id, event)

    async def _load_job(self, job_id: str) -> Optional[DownloadJob]:
        # Fallback loader: scan the queued+in_progress rows since T1 DB API
        # does not expose a direct get_by_id; use next_pending repeatedly only
        # as a last resort. We rely on upsert order: the queued row exists.
        import json
        async with self._db.conn.execute(
            "SELECT * FROM download_jobs WHERE id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return DownloadJob(
            id=row["id"],
            channel_id=row["channel_id"],
            author=row["author"],
            title=row["title"],
            message_ids=json.loads(row["message_ids"] or "[]"),
            status=row["status"],
            current_index=row["current_index"],
            total=row["total"],
            current_pct=row["current_pct"],
            overall_pct=row["overall_pct"],
            destination_dir=row["destination_dir"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


# ---------------------------------------------------------------------------
# Sync queue (channel scans)
# ---------------------------------------------------------------------------


class SyncQueue:
    """Dedicated FIFO + single worker for channel sync jobs.

    Jobs are kept in memory (not persisted) since a scan is cheap to re-run.
    Each job is a plain dict: ``{id, channel, limit}``.
    """

    def __init__(self, bus: JobEventBus, handler: SyncHandler) -> None:
        self._bus = bus
        self._handler = handler
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._worker: Optional[asyncio.Task[None]] = None
        self._current_id: Optional[str] = None
        self._current_task: Optional[asyncio.Task[None]] = None
        self._cancelled: set[str] = set()
        # Live snapshot of in-flight + queued sync jobs, keyed by job_id.
        # Survives page reloads/tab switches — consumed by GET /syncs/active.
        self._states: Dict[str, dict] = {}

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker(), name="sync-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._worker = None

    async def enqueue(self, channel: str, limit: Optional[int] = None) -> str:
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "channel": channel, "limit": limit}
        self._states[job_id] = {
            "job_id": job_id,
            "channel": channel,
            "status": "queued",
            "scanned": 0,
            "total": None,
            "progress": None,
            "message": "queued",
            "new": 0,
            "error": None,
            "elapsed_s": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._queue.put(job)
        await self._bus.publish(
            job_id, {"type": "queued", "data": {"job_id": job_id, "channel": channel}}
        )
        return job_id

    async def cancel(self, job_id: str) -> bool:
        self._cancelled.add(job_id)
        if self._current_id == job_id and self._current_task is not None:
            self._current_task.cancel()
            return True
        await self._bus.publish(job_id, {"type": "cancelled", "data": {"job_id": job_id}})
        await self._bus.close(job_id)
        self._states.pop(job_id, None)
        return True

    async def publish(self, job_id: str, event: dict) -> None:
        # Mirror the event into the live state snapshot so a newly-mounted
        # client can rehydrate without waiting for the next SSE tick.
        state = self._states.get(job_id)
        if state is not None:
            etype = event.get("type")
            data = event.get("data") or {}
            if etype == "progress":
                state["status"] = "running"
                if "scanned" in data:
                    state["scanned"] = data["scanned"]
                if "total" in data and data["total"] is not None:
                    state["total"] = data["total"]
                if "message" in data:
                    state["message"] = data["message"]
                if state.get("total"):
                    try:
                        state["progress"] = min(
                            100.0, (state["scanned"] / state["total"]) * 100.0
                        )
                    except Exception:  # noqa: BLE001
                        pass
            elif etype == "done":
                state["status"] = "done"
                state["scanned"] = data.get("scanned", state["scanned"])
                state["new"] = data.get("new", state["new"])
                state["elapsed_s"] = data.get("elapsed_s")
                state["progress"] = 100.0
                state["message"] = None
            elif etype == "error":
                state["status"] = "failed"
                state["error"] = data.get("message") or "error"
            elif etype == "cancelled":
                state["status"] = "cancelled"
        await self._bus.publish(job_id, event)

    def list_active(self) -> List[dict]:
        """Return snapshot of all non-terminal sync jobs (queued/running)."""
        return [
            dict(s) for s in self._states.values()
            if s.get("status") in ("queued", "running")
        ]

    def get_state(self, job_id: str) -> Optional[dict]:
        s = self._states.get(job_id)
        return dict(s) if s is not None else None

    async def _run_worker(self) -> None:
        while True:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                return
            if job["id"] in self._cancelled:
                self._cancelled.discard(job["id"])
                continue
            await self._process(job)

    async def _process(self, job: dict) -> None:
        job_id = job["id"]
        self._current_id = job_id
        coro = self._handler(job, self)
        self._current_task = asyncio.create_task(coro, name=f"sync-{job_id}")
        try:
            await self._current_task
        except asyncio.CancelledError:
            await self._bus.publish(job_id, {"type": "cancelled", "data": {"job_id": job_id}})
        except Exception as exc:  # noqa: BLE001
            log.exception("sync job %s failed", job_id)
            await self._bus.publish(job_id, {"type": "error", "data": {"message": str(exc)}})
        finally:
            self._current_id = None
            self._current_task = None
            self._cancelled.discard(job_id)
            await self._bus.close(job_id)
            asyncio.create_task(self._evict_state(job_id, delay=60.0))

    async def _evict_state(self, job_id: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        self._states.pop(job_id, None)


# ---------------------------------------------------------------------------
# Default handlers wired in app.py
# ---------------------------------------------------------------------------


LIBRARY_ROOT_ENV = "TG_LIBRARY_ROOT"
DEFAULT_LIBRARY_ROOT = "/media"


def library_root() -> Path:
    import os
    return Path(os.environ.get(LIBRARY_ROOT_ENV) or DEFAULT_LIBRARY_ROOT)


def build_download_handler(
    service_provider: Callable[[], Any],
    db: Database,
    downloader: Optional[Downloader] = None,
) -> DownloadHandler:
    """Factory returning the coroutine used by :class:`DownloadQueue`.

    ``service_provider`` returns the current :class:`TelegramService` (so we
    can hot-swap it when credentials change). ``downloader`` can be injected
    for tests.
    """
    dl = downloader or Downloader()

    async def handler(job: DownloadJob, queue: DownloadQueue) -> None:
        total = max(1, int(job.total))
        await db.mark_download_status(
            job.id, "in_progress", current_index=0, current_pct=0.0, overall_pct=0.0
        )
        await queue.publish(
            job.id,
            {
                "type": "progress",
                "data": {
                    "file_index": 0,
                    "total_files": total,
                    "current_pct": 0.0,
                    "overall_pct": 0.0,
                    "message": "starting",
                },
            },
        )

        svc = service_provider()
        client = await svc.ensure_ready()

        # Resolve channel entity once (Telethon can't lookup bare numeric IDs).
        channel_ref: Any = job.channel_id
        try:
            async with db.conn.execute(
                "SELECT username FROM channels WHERE channel_id = ?",
                (job.channel_id,),
            ) as cur:
                row = await cur.fetchone()
            if row and row["username"]:
                channel_ref = await client.get_entity(row["username"])
        except Exception:  # noqa: BLE001
            log.warning("could not resolve entity for channel %s", job.channel_id)

        # Resolve per-message metadata from the DB (author/title/chapter/ext).
        for i, msg_id in enumerate(job.message_ids, start=1):
            media = await _fetch_media_row(db, job.channel_id, int(msg_id))
            author = (media.get("author") if media else None) or job.author
            title = (media.get("title") if media else None) or job.title
            chapter_num = (media.get("chapter_num") if media else None) or i
            ext = _guess_ext(media.get("filename") if media else None,
                              media.get("mime_type") if media else None)

            filename = chapter_filename(author, title, chapter_num, ext)
            folder = instructional_dirname(job.author, job.title)
            dest = library_root() / folder / filename

            def _cb(current: int, total_b: int, _i=i, _total=total) -> None:
                pct = (current / total_b * 100.0) if total_b else 0.0
                overall = ((_i - 1) / _total + pct / 100.0 / _total) * 100.0
                # Schedule a non-blocking publish.
                loop = asyncio.get_event_loop()
                loop.create_task(queue.publish(
                    job.id,
                    {
                        "type": "progress",
                        "data": {
                            "file_index": _i,
                            "total_files": _total,
                            "current_pct": round(pct, 2),
                            "overall_pct": round(overall, 2),
                            "message": filename,
                        },
                    },
                ))

            saved_path, size = await dl.download(
                client, channel_ref, int(msg_id), str(dest), progress_cb=_cb
            )
            # Mark media as downloaded.
            try:
                await db.conn.execute(
                    "UPDATE media SET downloaded_path = ? WHERE channel_id = ? AND message_id = ?",
                    (saved_path, job.channel_id, int(msg_id)),
                )
                await db.conn.commit()
            except Exception:  # noqa: BLE001
                log.exception("failed to mark downloaded_path")

            overall = (i / total) * 100.0
            await db.mark_download_status(
                job.id, "in_progress",
                current_index=i, current_pct=100.0, overall_pct=overall,
            )

        await db.mark_download_status(
            job.id, "done", current_index=total, current_pct=100.0, overall_pct=100.0
        )
        await queue.publish(
            job.id,
            {
                "type": "done",
                "data": {
                    "job_id": job.id,
                    "destination_dir": job.destination_dir,
                    "total_files": total,
                },
            },
        )

    return handler


async def _fetch_media_row(db: Database, channel_id: str, message_id: int) -> Optional[dict]:
    async with db.conn.execute(
        "SELECT author, title, chapter_num, filename, mime_type FROM media "
        "WHERE channel_id = ? AND message_id = ?",
        (channel_id, int(message_id)),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    return {
        "author": row["author"],
        "title": row["title"],
        "chapter_num": row["chapter_num"],
        "filename": row["filename"],
        "mime_type": row["mime_type"],
    }


def _guess_ext(filename: Optional[str], mime: Optional[str]) -> str:
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[-1].lower()
    if mime:
        m = mime.lower()
        if m == "video/mp4":
            return ".mp4"
        if m == "video/x-matroska":
            return ".mkv"
        if m.startswith("video/"):
            return "." + m.split("/", 1)[1]
    return ".mp4"


def build_sync_handler(
    service_provider: Callable[[], Any],
    db: Database,
    browser: Optional[ChannelBrowser] = None,
) -> SyncHandler:
    """Factory for channel-scan jobs."""
    br = browser or ChannelBrowser(db)
    from .parser import parse_caption  # avoid import cycle at module top
    from .thumbnails import ensure_thumbnail

    async def handler(job: dict, queue: SyncQueue) -> None:
        channel = job["channel"]
        limit = job.get("limit")  # None = no cap (full scan)
        job_id = job["id"]
        svc = service_provider()
        client = await svc.ensure_ready()

        scanned = 0
        new = 0
        parser_hits = 0
        parser_misses = 0
        started = datetime.now(timezone.utc)

        await queue.publish(
            job_id,
            {"type": "progress", "data": {"scanned": 0, "channel": channel, "message": "starting"}},
        )

        async for item in br.iter_channel_media(client, channel, limit=(int(limit) if limit else None)):
            scanned += 1
            caption = item.get("caption") or ""
            parsed = parse_caption(caption) if caption else None
            author = parsed.author if parsed else None
            title = parsed.title if parsed else None
            chapter = parsed.chapter_num if parsed else None
            if parsed and parsed.confidence >= 0.5:
                parser_hits += 1
            else:
                parser_misses += 1

            from .models import MediaItem
            media = MediaItem(
                channel_id=item["channel_id"],
                message_id=int(item["message_id"]),
                caption=caption or None,
                filename=item.get("filename"),
                size_bytes=int(item.get("size_bytes") or 0),
                mime_type=item.get("mime_type"),
                date=item.get("date") or datetime.now(timezone.utc),
                author=author,
                title=title,
                chapter_num=chapter,
            )
            try:
                await db.upsert_media(media)
                new += 1
            except Exception:  # noqa: BLE001
                log.exception("upsert_media failed")

            # Best-effort thumbnail — never abort the sync on failure.
            raw_msg = item.get("_msg")
            if raw_msg is not None:
                try:
                    thumb_path = await ensure_thumbnail(
                        client, raw_msg, media.channel_id, media.message_id
                    )
                    if thumb_path:
                        try:
                            await db.set_thumbnail_path(
                                media.channel_id, media.message_id, thumb_path
                            )
                        except Exception:  # noqa: BLE001
                            log.warning(
                                "set_thumbnail_path failed for %s/%s",
                                media.channel_id, media.message_id,
                            )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "thumbnail step raised for %s/%s: %s",
                        media.channel_id, media.message_id, exc,
                    )

            if scanned % 10 == 0:
                await queue.publish(
                    job_id,
                    {
                        "type": "progress",
                        "data": {
                            "scanned": scanned,
                            "channel": channel,
                            "parser_hits": parser_hits,
                            "parser_misses": parser_misses,
                        },
                    },
                )

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        await queue.publish(
            job_id,
            {
                "type": "done",
                "data": {
                    "job_id": job_id,
                    "channel": channel,
                    "scanned": scanned,
                    "new": new,
                    "parser_hits": parser_hits,
                    "parser_misses": parser_misses,
                    "elapsed_s": elapsed,
                },
            },
        )

    return handler
