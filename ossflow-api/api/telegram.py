"""Telegram fetcher proxy/orchestrator router.

Bridges processor-api with the ``telegram-fetcher`` service (port 8004).

Responsibilities:
- Thin HTTP proxy: validate inputs, translate to backend call, forward response.
- Map backend 4xx to client (status + body). httpx errors -> 502. Timeout -> 504.
- Proxy SSE streams (``/events``) via chunked ``StreamingResponse``.

This module NEVER imports ``telegram_fetcher``; communication is HTTP only.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/telegram", tags=["telegram"])


# ---------------------------------------------------------------------------
# Background-job tracking (integration with Dashboard "Jobs activos")
# ---------------------------------------------------------------------------

async def _track_telegram_job(
    *,
    sse_url: str,
    job_kind: str,
) -> dict:
    """Subscribe to a telegram-fetcher SSE stream and drive a background_jobs
    entry so the processor-api dashboard can display it next to cleanups /
    duplicates scans.

    Returns the final event dict (e.g. ``done`` payload) or ``{}``.
    """
    import json as _json
    last_event: dict = {}
    try:
        async with httpx.AsyncClient(timeout=SSE_TIMEOUT) as client:
            async with client.stream("GET", sse_url) as r:
                if r.status_code >= 400:
                    raise RuntimeError(f"{job_kind} sse error: {r.status_code}")
                buf = b""
                async for chunk in r.aiter_bytes():
                    if not chunk:
                        continue
                    buf += chunk
                    while b"\n\n" in buf:
                        raw, buf = buf.split(b"\n\n", 1)
                        event_type = "message"
                        data_txt = ""
                        for line in raw.splitlines():
                            if line.startswith(b"event:"):
                                event_type = line.split(b":", 1)[1].strip().decode()
                            elif line.startswith(b"data:"):
                                data_txt += line.split(b":", 1)[1].strip().decode()
                        if not data_txt:
                            continue
                        try:
                            payload = _json.loads(data_txt)
                        except ValueError:
                            continue
                        last_event = payload
                        if event_type in ("done", "error", "cancelled"):
                            return payload
    except httpx.HTTPError as exc:
        log.warning("telegram tracker: backend unreachable: %s", exc)
    return last_event


def _register_bg_job(kind: str, job_id: str, sse_path: str, params: dict) -> None:
    """Register a ``telegram_sync`` / ``telegram_download`` entry in the
    processor-api background-jobs registry.

    The coroutine streams the backend SSE to mark the job ``completed`` once
    the backend emits ``done`` (or ``failed`` on error/cancel).
    """
    try:
        from api.background_jobs import registry  # local import to avoid cycle
    except Exception:  # noqa: BLE001
        log.debug("background_jobs registry unavailable; skipping tracker")
        return

    sse_url = f"{_backend_base()}{sse_path}"

    async def factory(update_progress):
        update_progress(0.0, f"waiting for {kind}")
        payload = await _track_telegram_job(sse_url=sse_url, job_kind=kind)
        # Map progress + message for dashboard display.
        evt_type = payload.get("type")
        if evt_type == "error" or payload.get("status") in ("failed",):
            raise RuntimeError(payload.get("data", {}).get("message") or "failed")
        update_progress(100.0, f"{kind} done")
        return {
            "backend_job_id": job_id,
            "last_event": payload,
        }

    enriched = dict(params or {})
    enriched["backend_job_id"] = job_id
    enriched["sse_url"] = sse_path
    try:
        registry.submit(kind, factory, enriched)
    except Exception:  # noqa: BLE001
        log.exception("failed to register %s background job", kind)

DEFAULT_TIMEOUT = 30.0
SSE_TIMEOUT = httpx.Timeout(None, connect=10.0)  # long-lived stream


def _backend_base() -> str:
    # telegram-fetcher routers mount under "/telegram" prefix, so include it
    # here once instead of repeating "/telegram" in every endpoint path.
    base = os.environ.get(
        "TELEGRAM_FETCHER_URL", "http://telegram-fetcher:8004"
    ).rstrip("/")
    return f"{base}/telegram"


# ---------------------------------------------------------------------------
# Internal helpers: one-shot JSON proxy
# ---------------------------------------------------------------------------

async def _proxy_json(
    method: str,
    path: str,
    *,
    json_body: Any = None,
    params: dict[str, Any] | None = None,
) -> JSONResponse:
    url = f"{_backend_base()}{path}"
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.request(method, url, json=json_body, params=params)
    except httpx.TimeoutException as exc:
        log.warning("telegram backend timeout %s %s: %s", method, path, exc)
        raise HTTPException(status_code=504, detail=f"backend timeout: {exc}")
    except httpx.HTTPError as exc:
        log.warning("telegram backend unreachable %s %s: %s", method, path, exc)
        raise HTTPException(status_code=502, detail=f"backend unreachable: {exc}")

    # Try to decode JSON regardless of status
    try:
        payload = r.json()
    except ValueError:
        payload = {"detail": r.text}

    if r.status_code >= 400:
        # Forward backend 4xx/5xx status + body
        return JSONResponse(payload, status_code=r.status_code)
    return JSONResponse(payload, status_code=r.status_code)


# ---------------------------------------------------------------------------
# SSE proxy
# ---------------------------------------------------------------------------

async def _sse_proxy(path: str) -> StreamingResponse:
    url = f"{_backend_base()}{path}"

    async def gen() -> AsyncIterator[bytes]:
        try:
            async with httpx.AsyncClient(timeout=SSE_TIMEOUT) as client:
                async with client.stream("GET", url) as r:
                    if r.status_code >= 400:
                        body = await r.aread()
                        yield (
                            f"event: error\ndata: backend {r.status_code}: "
                            f"{body.decode('utf-8', errors='replace')}\n\n"
                        ).encode("utf-8")
                        return
                    async for chunk in r.aiter_bytes():
                        if chunk:
                            yield chunk
        except httpx.HTTPError as exc:
            yield f"event: error\ndata: backend unreachable: {exc}\n\n".encode(
                "utf-8"
            )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Body helpers
# ---------------------------------------------------------------------------

async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="body must be a JSON object")
    return body


def _require_str(body: dict[str, Any], key: str) -> str:
    val = body.get(key)
    if not isinstance(val, str) or not val.strip():
        raise HTTPException(
            status_code=422, detail=f"'{key}' must be a non-empty string"
        )
    return val


# ---------------------------------------------------------------------------
# Status + Auth
# ---------------------------------------------------------------------------

@router.get("/status")
async def get_status():
    return await _proxy_json("GET", "/status")


@router.post("/auth/send-code")
async def auth_send_code(request: Request):
    body = await _json_body(request)
    phone = _require_str(body, "phone")
    return await _proxy_json("POST", "/auth/send-code", json_body={"phone": phone})


@router.post("/auth/sign-in")
async def auth_sign_in(request: Request):
    body = await _json_body(request)
    phone = _require_str(body, "phone")
    code = _require_str(body, "code")
    payload: dict[str, Any] = {"phone": phone, "code": code}
    # forward optional phone_code_hash if present
    if isinstance(body.get("phone_code_hash"), str):
        payload["phone_code_hash"] = body["phone_code_hash"]
    return await _proxy_json("POST", "/auth/sign-in", json_body=payload)


@router.post("/auth/2fa")
async def auth_2fa(request: Request):
    body = await _json_body(request)
    password = _require_str(body, "password")
    return await _proxy_json("POST", "/auth/2fa", json_body={"password": password})


@router.post("/auth/logout")
async def auth_logout():
    return await _proxy_json("POST", "/auth/logout")


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

@router.get("/channels")
async def list_channels():
    return await _proxy_json("GET", "/channels")


@router.post("/channels")
async def add_channel(request: Request):
    body = await _json_body(request)
    username = _require_str(body, "username")
    return await _proxy_json("POST", "/channels", json_body={"username": username})


@router.patch("/channels/{channel_id}")
async def update_channel(channel_id: str, request: Request):
    body = await _json_body(request)
    title = _require_str(body, "title")
    return await _proxy_json(
        "PATCH", f"/channels/{channel_id}", json_body={"title": title}
    )


@router.delete("/channels/{channel_id}")
async def delete_channel(channel_id: str):
    return await _proxy_json("DELETE", f"/channels/{channel_id}")


@router.get("/syncs/active")
async def list_active_syncs():
    return await _proxy_json("GET", "/syncs/active")


@router.post("/channels/{username}/sync")
async def sync_channel(username: str, request: Request):
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
            raise HTTPException(status_code=422, detail="'limit' must be int or null")
        payload["limit"] = limit
    resp = await _proxy_json(
        "POST", f"/channels/{username}/sync", json_body=payload
    )
    # Register dashboard background job on success.
    try:
        if 200 <= resp.status_code < 300:
            import json as _json
            body_bytes = resp.body if hasattr(resp, "body") else b""
            data = _json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            job_id = data.get("job_id") or data.get("id")
            if job_id:
                _register_bg_job(
                    "telegram_sync",
                    job_id,
                    f"/channels/{username}/sync/{job_id}/events",
                    {"username": username, "channel": username},
                )
    except Exception:  # noqa: BLE001
        log.exception("failed to track telegram_sync")
    return resp


@router.get("/channels/{username}/sync/{job_id}/events")
async def sync_channel_events(username: str, job_id: str):
    return await _sse_proxy(f"/channels/{username}/sync/{job_id}/events")


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
    data = await _proxy_json("GET", "/media", params=params)
    if view == "by_author" and isinstance(data, dict):
        data = _apply_author_aliases(data)
    return data


def _apply_author_aliases(data: dict) -> dict:
    """Merge author groups according to ``author_aliases`` from settings.

    settings.author_aliases maps ``"raw name"`` -> ``"canonical name"``.
    Groups whose author matches a key get rewritten and merged into the
    canonical bucket (instructionals concatenated, counters summed).
    """
    from api.settings import load_settings

    try:
        aliases_raw = load_settings().get("author_aliases") or {}
    except Exception:  # noqa: BLE001
        return data
    if not aliases_raw:
        return data

    # Build a lookup normalized on lowercased-stripped key so user doesn't
    # have to get casing exactly right.
    lookup = {str(k).strip().lower(): str(v).strip() for k, v in aliases_raw.items() if k and v}
    if not lookup:
        return data

    authors = data.get("authors")
    if not isinstance(authors, list):
        return data

    merged: dict[str, dict] = {}
    order: list[str] = []
    for a in authors:
        if not isinstance(a, dict):
            continue
        name = str(a.get("name") or "").strip()
        canonical = lookup.get(name.lower(), name)
        bucket = merged.get(canonical)
        if bucket is None:
            bucket = {**a, "name": canonical, "instructionals": list(a.get("instructionals") or [])}
            merged[canonical] = bucket
            order.append(canonical)
        else:
            bucket["instructionals"].extend(a.get("instructionals") or [])

    out_authors = [merged[k] for k in order]
    out_authors.sort(key=lambda x: str(x.get("name") or "").lower())
    return {**data, "authors": out_authors}


@router.get("/media/{channel_id}/{message_id}/thumbnail")
async def get_media_thumbnail(channel_id: str, message_id: str):
    """Proxy the thumbnail binary from telegram-fetcher.

    Streams bytes through so we don't buffer every frame for every card.
    """
    url = f"{_backend_base()}/media/{channel_id}/{message_id}/thumbnail"
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(url)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail=f"backend timeout: {exc}")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"backend unreachable: {exc}")
    if r.status_code == 404:
        raise HTTPException(status_code=404, detail="thumbnail not available")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return Response(
        content=r.content,
        media_type=r.headers.get("content-type", "image/jpeg"),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.put("/media/{channel_id}/{message_id}")
async def put_media_metadata(channel_id: str, message_id: str, request: Request):
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
            detail="body must include at least one of author/title/chapter_num",
        )
    return await _proxy_json(
        "PUT", f"/media/{channel_id}/{message_id}", json_body=payload
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

@router.post("/download")
async def start_download(request: Request):
    body = await _json_body(request)
    channel_id = body.get("channel_id")
    if not isinstance(channel_id, (str, int)) or (
        isinstance(channel_id, str) and not channel_id.strip()
    ):
        raise HTTPException(
            status_code=422, detail="'channel_id' must be string or int"
        )
    author = _require_str(body, "author")
    title = _require_str(body, "title")
    payload = {"channel_id": channel_id, "author": author, "title": title}
    resp = await _proxy_json("POST", "/download", json_body=payload)
    try:
        if 200 <= resp.status_code < 300:
            import json as _json
            body_bytes = resp.body if hasattr(resp, "body") else b""
            data = _json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            job_id = data.get("job_id") or data.get("id")
            if job_id:
                _register_bg_job(
                    "telegram_download",
                    job_id,
                    f"/download/{job_id}/events",
                    {"author": author, "title": title, "channel_id": channel_id},
                )
    except Exception:  # noqa: BLE001
        log.exception("failed to track telegram_download")
    return resp


@router.get("/download/{job_id}/events")
async def download_events(job_id: str):
    return await _sse_proxy(f"/download/{job_id}/events")


@router.post("/download/{job_id}/cancel")
async def cancel_download(job_id: str):
    return await _proxy_json("POST", f"/download/{job_id}/cancel")


@router.get("/download/jobs")
async def list_download_jobs(status: str | None = None):
    params: dict[str, Any] = {}
    if status is not None:
        params["status"] = status
    return await _proxy_json("GET", "/download/jobs", params=params)
