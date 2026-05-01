"""Endpoints HTTP del módulo telegram.

Reproduce el contrato del antiguo ``api/telegram.py``: mismos paths,
mismos métodos, mismos payloads, misma validación manual sobre el
body crudo. La lógica vive en ``TelegramService``; este router sólo
parsea inputs y delega.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from .dependencies import get_telegram_service
from .schemas import require_str
from .service import TelegramService

router = APIRouter(prefix="/api/telegram", tags=["telegram"])


# ---------------------------------------------------------------------------
# Body helpers
# ---------------------------------------------------------------------------


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=422, detail="body must be a JSON object"
        )
    return body


# ---------------------------------------------------------------------------
# Status + Auth
# ---------------------------------------------------------------------------


@router.get("/status")
async def get_status(svc: TelegramService = Depends(get_telegram_service)):
    return await svc.get_status()


@router.post("/auth/send-code")
async def auth_send_code(
    request: Request, svc: TelegramService = Depends(get_telegram_service)
):
    body = await _json_body(request)
    phone = require_str(body, "phone")
    return await svc.auth_send_code(phone)


@router.post("/auth/sign-in")
async def auth_sign_in(
    request: Request, svc: TelegramService = Depends(get_telegram_service)
):
    body = await _json_body(request)
    phone = require_str(body, "phone")
    code = require_str(body, "code")
    payload: dict[str, Any] = {"phone": phone, "code": code}
    if isinstance(body.get("phone_code_hash"), str):
        payload["phone_code_hash"] = body["phone_code_hash"]
    return await svc.auth_sign_in(payload)


@router.post("/auth/2fa")
async def auth_2fa(
    request: Request, svc: TelegramService = Depends(get_telegram_service)
):
    body = await _json_body(request)
    password = require_str(body, "password")
    return await svc.auth_2fa(password)


@router.post("/auth/logout")
async def auth_logout(svc: TelegramService = Depends(get_telegram_service)):
    return await svc.auth_logout()


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


@router.get("/channels")
async def list_channels(svc: TelegramService = Depends(get_telegram_service)):
    return await svc.list_channels()


@router.post("/channels")
async def add_channel(
    request: Request, svc: TelegramService = Depends(get_telegram_service)
):
    body = await _json_body(request)
    username = require_str(body, "username")
    return await svc.add_channel(username)


@router.patch("/channels/{channel_id}")
async def update_channel(
    channel_id: str,
    request: Request,
    svc: TelegramService = Depends(get_telegram_service),
):
    body = await _json_body(request)
    title = require_str(body, "title")
    return await svc.update_channel(channel_id, title)


@router.delete("/channels/{channel_id}")
async def delete_channel(
    channel_id: str, svc: TelegramService = Depends(get_telegram_service)
):
    return await svc.delete_channel(channel_id)


@router.get("/syncs/active")
async def list_active_syncs(
    svc: TelegramService = Depends(get_telegram_service),
):
    return await svc.list_active_syncs()


@router.post("/channels/{username}/sync")
async def sync_channel(
    username: str,
    request: Request,
    svc: TelegramService = Depends(get_telegram_service),
):
    try:
        body = await request.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    payload: dict[str, Any] = {}
    if "limit" in body:
        limit = body["limit"]
        if limit is not None and not isinstance(limit, int):
            raise HTTPException(
                status_code=422, detail="'limit' must be int or null"
            )
        payload["limit"] = limit
    return await svc.sync_channel(username, payload)


@router.get("/channels/{username}/sync/{job_id}/events")
async def sync_channel_events(
    username: str,
    job_id: str,
    svc: TelegramService = Depends(get_telegram_service),
):
    return await svc.sync_channel_events(username, job_id)


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


@router.get("/media")
async def list_media(
    channel: str | None = None,
    view: str | None = None,
    search: str | None = None,
    page: int | None = None,
    page_size: int | None = None,
    svc: TelegramService = Depends(get_telegram_service),
):
    params: dict[str, Any] = {}
    if channel is not None:
        params["channel"] = channel
    if view is not None:
        if view not in {"chronological", "by_author"}:
            raise HTTPException(
                status_code=422,
                detail="view must be 'chronological' or 'by_author'",
            )
        params["view"] = view
    if search is not None:
        params["search"] = search
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    return await svc.list_media(params)


@router.get("/media/{channel_id}/{message_id}/thumbnail")
async def get_media_thumbnail(
    channel_id: str,
    message_id: str,
    svc: TelegramService = Depends(get_telegram_service),
):
    """Reenvía el thumbnail binario desde telegram-fetcher."""
    return await svc.get_media_thumbnail(channel_id, message_id)


@router.put("/media/{channel_id}/{message_id}")
async def put_media_metadata(
    channel_id: str,
    message_id: str,
    request: Request,
    svc: TelegramService = Depends(get_telegram_service),
):
    body = await _json_body(request)
    payload: dict[str, Any] = {}
    if "author" in body:
        if not isinstance(body["author"], str):
            raise HTTPException(status_code=422, detail="'author' must be string")
        payload["author"] = body["author"]
    if "title" in body:
        if not isinstance(body["title"], str):
            raise HTTPException(status_code=422, detail="'title' must be string")
        payload["title"] = body["title"]
    if "chapter_num" in body:
        cn = body["chapter_num"]
        if cn is not None and not isinstance(cn, int):
            raise HTTPException(
                status_code=422, detail="'chapter_num' must be int or null"
            )
        payload["chapter_num"] = cn
    if not payload:
        raise HTTPException(
            status_code=422,
            detail=(
                "body must include at least one of author/title/chapter_num"
            ),
        )
    return await svc.put_media_metadata(channel_id, message_id, payload)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@router.post("/download")
async def start_download(
    request: Request, svc: TelegramService = Depends(get_telegram_service)
):
    body = await _json_body(request)
    channel_id = body.get("channel_id")
    if not isinstance(channel_id, (str, int)) or (
        isinstance(channel_id, str) and not channel_id.strip()
    ):
        raise HTTPException(
            status_code=422, detail="'channel_id' must be string or int"
        )
    author = require_str(body, "author")
    title = require_str(body, "title")
    payload = {"channel_id": channel_id, "author": author, "title": title}
    return await svc.start_download(payload)


@router.get("/download/{job_id}/events")
async def download_events(
    job_id: str, svc: TelegramService = Depends(get_telegram_service)
):
    return await svc.download_events(job_id)


@router.post("/download/{job_id}/cancel")
async def cancel_download(
    job_id: str, svc: TelegramService = Depends(get_telegram_service)
):
    return await svc.cancel_download(job_id)


@router.get("/download/jobs")
async def list_download_jobs(
    status: str | None = None,
    svc: TelegramService = Depends(get_telegram_service),
):
    return await svc.list_download_jobs(status)
