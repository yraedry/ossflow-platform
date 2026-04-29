"""Download endpoints: enqueue, SSE events, cancel, list jobs."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..models import DownloadJob
from ..naming import instructional_dirname
from ..queue import library_root


log = logging.getLogger(__name__)


router = APIRouter(prefix="/telegram", tags=["telegram-download"])


class _DownloadBody(BaseModel):
    channel_id: str
    author: str
    title: str


@router.post("/download")
async def enqueue_download(body: _DownloadBody, request: Request) -> dict:
    db = request.app.state.db
    dq = request.app.state.download_queue

    # Resolve message_ids for this (channel_id, author, title), ordered by chapter.
    async with db.conn.execute(
        """
        SELECT message_id, chapter_num FROM media
        WHERE channel_id = ? AND author = ? AND title = ?
        ORDER BY COALESCE(chapter_num, message_id) ASC
        """,
        (body.channel_id, body.author, body.title),
    ) as cur:
        rows = await cur.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="no media matches (channel_id, author, title)")

    message_ids = [int(r["message_id"]) for r in rows]
    folder = instructional_dirname(body.author, body.title)
    dest = str(library_root() / folder)

    now = datetime.now(timezone.utc)
    job = DownloadJob(
        id=uuid.uuid4().hex,
        channel_id=body.channel_id,
        author=body.author,
        title=body.title,
        message_ids=message_ids,
        status="queued",
        total=len(message_ids),
        destination_dir=dest,
        created_at=now,
        updated_at=now,
    )
    await dq.enqueue(job)

    # Register completion hook for scan trigger (first hook only).
    completed_cb = request.app.state.download_completion_cb
    if completed_cb is not None:
        asyncio.create_task(completed_cb(job.id))
    return {"job_id": job.id}


@router.get("/download/{job_id}/events")
async def download_events(job_id: str, request: Request) -> StreamingResponse:
    bus = request.app.state.event_bus
    q = await bus.subscribe(job_id)

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if evt.get("type") == "__end__":
                    break
                payload = json.dumps(evt, default=str)
                yield f"event: {evt.get('type','message')}\ndata: {payload}\n\n"
        finally:
            await bus.unsubscribe(job_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/download/{job_id}/cancel")
async def cancel_download(job_id: str, request: Request) -> dict:
    dq = request.app.state.download_queue
    await dq.cancel(job_id)
    return {"ok": True}


@router.get("/download/jobs")
async def list_jobs(
    request: Request,
    status: Optional[str] = Query(default=None),
) -> list[dict]:
    db = request.app.state.db
    if status:
        sql = "SELECT * FROM download_jobs WHERE status = ? ORDER BY created_at DESC LIMIT 500"
        params = (status,)
    else:
        sql = "SELECT * FROM download_jobs ORDER BY created_at DESC LIMIT 500"
        params = ()
    async with db.conn.execute(sql, params) as cur:
        rows = await cur.fetchall()
    out: list[dict] = []
    for r in rows:
        out.append({
            "id": r["id"],
            "channel_id": r["channel_id"],
            "author": r["author"],
            "title": r["title"],
            "status": r["status"],
            "current_index": r["current_index"],
            "total": r["total"],
            "current_pct": r["current_pct"],
            "overall_pct": r["overall_pct"],
            "destination_dir": r["destination_dir"],
            "error": r["error"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })
    return out


async def trigger_library_scan(processor_api_url: str) -> None:
    """Fire-and-forget POST to processor-api library/scan. Timeout 2s, never raises."""
    url = f"{processor_api_url.rstrip('/')}/api/library/scan"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(url, json={})
    except Exception as exc:  # noqa: BLE001
        log.warning("library scan trigger failed: %s", exc)
