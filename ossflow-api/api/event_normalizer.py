"""Normalize backend SSE events into a stable internal contract.

Backends built on ``bjj_service_kit.events`` emit events shaped like::

    {"type": "log|progress|done|error", "data": {...}}

Legacy or custom backends may instead emit flat events like::

    {"status": "running", "progress": 0.5, "message": "..."}

Both shapes are accepted. ``normalize`` always returns a
``NormalizedEvent`` with stable keys downstream code can rely on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class NormalizedEvent:
    kind: str                      # "log" | "progress" | "done" | "error" | "unknown"
    status: str                    # "running" | "completed" | "failed" | "" ...
    progress: Optional[float]      # 0..100 if present, else None
    message: Optional[str]
    payload: dict                  # full data dict (result, extras, etc.)

    def __getitem__(self, key: str) -> Any:  # dict-like for convenience
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


_TERMINAL_KINDS = {"done", "error"}


def is_terminal(evt: NormalizedEvent) -> bool:
    return evt.kind in _TERMINAL_KINDS


def _coerce_progress(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    # Both 0..1 and 0..100 are accepted; normalize to 0..100
    return p * 100.0 if p <= 1.0 else p


def normalize(raw: dict) -> NormalizedEvent:
    """Map any backend event shape to a NormalizedEvent."""
    if not isinstance(raw, dict):
        return NormalizedEvent(kind="unknown", status="", progress=None,
                               message=None, payload={})

    # --- New bjj_service_kit contract: {"type": ..., "data": {...}} ---
    if "type" in raw and "data" in raw and isinstance(raw["data"], dict):
        typ = str(raw.get("type", "")).lower()
        data = raw["data"]

        if typ == "done":
            return NormalizedEvent(
                kind="done",
                status="completed",
                progress=100.0,
                message=data.get("message"),
                payload=data,
            )
        if typ == "error":
            msg = data.get("message") or data.get("error") or "backend error"
            return NormalizedEvent(
                kind="error",
                status="failed",
                progress=None,
                message=str(msg),
                payload=data,
            )
        if typ == "progress":
            raw_pct = (
                data.get("percent")
                if data.get("percent") is not None
                else data.get("progress")
                if data.get("progress") is not None
                else data.get("pct")
            )
            prog = _coerce_progress(raw_pct)
            return NormalizedEvent(
                kind="progress",
                status="running",
                progress=prog,
                message=data.get("message"),
                payload=data,
            )
        if typ == "log":
            msg = data.get("line") or data.get("message")
            if msg is None:
                msg = str(data) if data else None
            return NormalizedEvent(
                kind="log",
                status="running",
                progress=None,
                message=str(msg) if msg is not None else None,
                payload=data,
            )
        # Unknown type -> best-effort
        return NormalizedEvent(
            kind="unknown", status="", progress=None,
            message=data.get("message"), payload=data,
        )

    # --- Legacy flat contract: {"status": ..., "progress": ..., ...} ---
    status = str(raw.get("status", "")).lower()
    progress = _coerce_progress(raw.get("progress"))
    message = raw.get("message")
    if message is not None:
        message = str(message)

    if status in ("failed", "error"):
        return NormalizedEvent(
            kind="error",
            status="failed",
            progress=progress,
            message=message or str(raw.get("error") or "backend error"),
            payload=raw,
        )
    if status in ("done", "completed"):
        return NormalizedEvent(
            kind="done",
            status="completed",
            progress=100.0,
            message=message,
            payload=raw,
        )
    if status == "running" or progress is not None:
        return NormalizedEvent(
            kind="progress",
            status=status or "running",
            progress=progress,
            message=message,
            payload=raw,
        )
    return NormalizedEvent(
        kind="log",
        status=status,
        progress=progress,
        message=message,
        payload=raw,
    )
