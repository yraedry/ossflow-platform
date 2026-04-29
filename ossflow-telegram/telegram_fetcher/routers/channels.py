"""Channel CRUD + sync endpoints.

Channels live in SQLite only — add/edit/remove exclusively through the API so
the frontend can manage them without code edits. Adding a channel validates
it against Telegram via Telethon before persisting.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..browser import ChannelBrowser
from ..errors import ChannelNotFoundError, TelegramError


log = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["telegram-channels"])


class _SyncBody(BaseModel):
    limit: Optional[int] = None


class _AddChannelBody(BaseModel):
    username: str = Field(..., min_length=1)


class _PatchChannelBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


def _normalize_username(raw: str) -> str:
    v = (raw or "").strip()
    if v.startswith("https://t.me/"):
        v = v[len("https://t.me/"):]
    if v.startswith("t.me/"):
        v = v[len("t.me/"):]
    v = v.lstrip("@").strip("/")
    if not v:
        raise HTTPException(status_code=400, detail="username vacío")
    return v


@router.get("/channels")
async def list_channels(request: Request) -> list[dict]:
    db = request.app.state.db
    return await db.list_channels()


@router.post("/channels", status_code=201)
async def add_channel(body: _AddChannelBody, request: Request) -> dict:
    username = _normalize_username(body.username)
    db = request.app.state.db

    existing = await db.get_channel_by_username(username)
    if existing:
        raise HTTPException(status_code=409, detail="canal ya existe")

    tg = request.app.state.telegram_service
    try:
        client = await tg.ensure_ready()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail=f"telegram no autenticado: {exc}") from exc

    browser = ChannelBrowser(db=db)
    try:
        info = await browser.resolve_channel(client, username)
    except ChannelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"canal no encontrado: {username}") from exc
    except TelegramError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # resolve_channel already upserts with the resolved handle + title.
    channels = await db.list_channels()
    for ch in channels:
        if ch["channel_id"] == info["channel_id"]:
            return ch
    return {
        "channel_id": info["channel_id"],
        "username": info["username"],
        "title": info.get("title"),
        "last_sync_at": None,
        "last_message_id": None,
        "media_count": 0,
        "noforwards": bool(info.get("noforwards", False)),
    }


@router.patch("/channels/{channel_id}")
async def update_channel(channel_id: str, body: _PatchChannelBody, request: Request) -> dict:
    db = request.app.state.db
    ok = await db.update_channel_title(channel_id, body.title.strip())
    if not ok:
        raise HTTPException(status_code=404, detail="canal no encontrado")
    for ch in await db.list_channels():
        if ch["channel_id"] == channel_id:
            return ch
    raise HTTPException(status_code=404, detail="canal no encontrado")


@router.delete("/channels/{channel_id}", status_code=204)
async def delete_channel(channel_id: str, request: Request) -> None:
    db = request.app.state.db
    thumb_paths = await db.list_media_thumbnails(channel_id)
    existed = await db.delete_channel_cascade(channel_id)
    if not existed:
        raise HTTPException(status_code=404, detail="canal no encontrado")
    for p in thumb_paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            log.exception("no se pudo borrar thumbnail %s", p)
    return None


@router.get("/syncs/active")
async def list_active_syncs(request: Request) -> list[dict]:
    """Return in-flight / queued sync jobs so clients can rehydrate UI.

    Survives page reloads and tab switches (state lives in SyncQueue, not in
    any client). Terminal jobs (done/failed/cancelled) are retained briefly so
    a client that reconnects just after completion still sees the final state.
    """
    sync_queue = request.app.state.sync_queue
    if sync_queue is None:
        return []
    return sync_queue.list_active()


@router.post("/channels/{username}/sync")
async def sync_channel(username: str, body: _SyncBody, request: Request) -> dict:
    sync_queue = request.app.state.sync_queue
    job_id = await sync_queue.enqueue(username, limit=body.limit)
    return {"job_id": job_id}


@router.get("/channels/{username}/sync/{job_id}/events")
async def sync_events(username: str, job_id: str, request: Request) -> StreamingResponse:
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
