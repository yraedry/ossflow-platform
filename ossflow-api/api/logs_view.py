"""Centralized logs viewer for BJJ processor services.

Each backend (chapter-splitter/subtitle/dubbing) exposes a `/logs` endpoint
backed by an in-memory ring buffer (bjj_service_kit). This aggregator forwards
the request to the right backend over HTTP — no `docker` binary needed in the
processor-api container.

processor-api itself also installs the same ring buffer at startup, so its
own logs are served locally without any network hop.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Deque, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query

log = logging.getLogger(__name__)


class RingBufferHandler(logging.Handler):
    """In-memory log ring for this process."""

    def __init__(self, capacity: int = 2000) -> None:
        super().__init__(level=logging.DEBUG)
        self.buffer: Deque[dict] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append({
                "timestamp": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            })
        except Exception:  # pragma: no cover
            pass


def _install_ring_buffer() -> RingBufferHandler:
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, RingBufferHandler):
            return h
    handler = RingBufferHandler()
    if root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    return handler

router = APIRouter(prefix="/api/logs", tags=["logs"])

ALLOWED_LEVELS = {"INFO", "WARN", "WARNING", "ERROR", "DEBUG", "ALL"}

# service name -> backend base URL (None = local ring buffer, this process)
SERVICE_URLS: dict[str, Optional[str]] = {
    "processor-api": None,
    "chapter-splitter": os.environ.get("SPLITTER_URL", "http://chapter-splitter:8001"),
    "subtitle-generator": os.environ.get("SUBS_URL", "http://subtitle-generator:8002"),
    "dubbing-generator": os.environ.get("DUBBING_URL", "http://dubbing-generator:8003"),
    "telegram-fetcher": os.environ.get("TELEGRAM_FETCHER_URL", "http://telegram-fetcher:8004"),
}

# Install the ring buffer for processor-api itself so its own logs are visible.
_LOCAL_BUFFER: RingBufferHandler = _install_ring_buffer()


def _normalize_level(level: Optional[str]) -> Optional[str]:
    if not level:
        return None
    lvl = level.upper()
    if lvl == "ALL":
        return None
    if lvl == "WARN":
        lvl = "WARNING"
    if lvl not in {"INFO", "WARNING", "ERROR", "DEBUG"}:
        return None
    return lvl


def _local_lines(level: Optional[str], tail: int) -> list[dict]:
    buf = list(_LOCAL_BUFFER.buffer)
    if level:
        buf = [r for r in buf if r.get("level") == level]
    if tail > 0:
        buf = buf[-tail:]
    # Shape matches backends: {timestamp, level, message}.
    return [
        {"timestamp": r.get("timestamp"), "level": r.get("level"), "message": r.get("message")}
        for r in buf
    ]


@router.get("/")
def get_logs(
    service: str = Query(..., description="Service name"),
    level: Optional[str] = Query(None, description="Filter by level (INFO/WARN/ERROR/DEBUG/ALL)"),
    tail: int = Query(500, ge=1, le=5000),
):
    if service not in SERVICE_URLS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown service '{service}'. Allowed: {sorted(SERVICE_URLS)}",
        )
    if level is not None and level.upper() not in ALLOWED_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid level '{level}'. Allowed: INFO, WARN, ERROR, DEBUG, ALL",
        )
    normalized = _normalize_level(level)

    base_url = SERVICE_URLS[service]
    if base_url is None:
        return {"service": service, "lines": _local_lines(normalized, tail), "truncated": False}

    # Remote backend — ask it for its ring buffer.
    try:
        resp = httpx.get(
            f"{base_url}/logs",
            params={"level": normalized or "ALL", "tail": tail},
            timeout=3.0,
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"backend {service} unreachable: {exc}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"backend {service} returned {resp.status_code}")
    data = resp.json() or {}
    return {
        "service": service,
        "lines": data.get("lines", []),
        "truncated": bool(data.get("truncated", False)),
    }
