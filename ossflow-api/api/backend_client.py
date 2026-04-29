"""HTTP client for backend microservices (splitter, subs, dubbing).

Single responsibility: talk HTTP + parse SSE. Nothing else.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Optional

import httpx

from api.event_normalizer import NormalizedEvent, is_terminal, normalize

log = logging.getLogger(__name__)

RUN_TIMEOUT = 10.0
STREAM_RECONNECT_DELAY = 2.0
# SSE streams: el backend envía heartbeat cada ~15 s. Si pasan 120 s
# sin ningún dato el backend está colgado → reconectamos. connect/write
# son rápidos (<10 s). pool es el slot del connection pool; alto para
# no bloquear.
_STREAM_TIMEOUT = httpx.Timeout(
    connect=10.0, read=120.0, write=10.0, pool=30.0,
)


class BackendError(RuntimeError):
    """Raised when a backend returns an error response."""


class BackendClient:
    """Async HTTP client for a single backend microservice."""

    def __init__(self, base_url: str, *, run_timeout: float = RUN_TIMEOUT) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self._run_timeout = run_timeout

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._run_timeout) as client:
            r = await client.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()

    async def run(self, payload: dict[str, Any]) -> str:
        """POST /run with payload, return job_id from response."""
        return await self._post_job("/run", payload)

    async def run_oracle(self, payload: dict[str, Any]) -> str:
        """POST /run-oracle with payload, return job_id from response."""
        return await self._post_job("/run-oracle", payload)

    async def _post_job(self, path: str, payload: dict[str, Any]) -> str:
        async with httpx.AsyncClient(timeout=self._run_timeout) as client:
            r = await client.post(f"{self.base_url}{path}", json=payload)
            if r.status_code >= 400:
                raise BackendError(f"{r.status_code}: {r.text}")
            data = r.json()
            job_id = data.get("job_id") or data.get("id")
            if not job_id:
                raise BackendError(f"No job_id in response: {data}")
            return job_id

    async def stream(
        self, job_id: str, *, max_reconnects: int = 3
    ) -> AsyncIterator[NormalizedEvent]:
        """Stream SSE events from /events/{job_id}, yield NormalizedEvent.

        Accepts both the bjj_service_kit contract (``{"type","data"}``)
        and the legacy flat contract (``{"status","progress",...}``).
        Reconnects on transient disconnect up to ``max_reconnects`` times.
        Stops when a terminal (done/error) event arrives.

        404 handling: if a reconnect lands on a 404 *after* we already
        saw at least one event, the job almost certainly completed and
        was reaped from the backend's in-memory registry between our
        reconnect attempts (or the backend itself restarted to free
        VRAM). Treat that as "stream closed cleanly" rather than a
        backend error — the alternative is marking a successful step
        as FAILED, which we observed on long dubbing jobs (~70 min) where
        the synthesis→mixing→mux transitions can sit silent for >120 s
        and trip the read timeout.

        First-attempt 404 (no events yet seen) keeps raising — that's a
        genuine "job_id not found" and likely caller bug.
        """
        url = f"{self.base_url}/events/{job_id}"
        attempts = 0
        seen_any_event = False
        while True:
            try:
                async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
                    async with client.stream("GET", url) as resp:
                        if resp.status_code == 404 and seen_any_event:
                            log.info(
                                "SSE 404 on reconnect for %s — job likely "
                                "completed and reaped, treating as clean close",
                                url,
                            )
                            return
                        if resp.status_code >= 400:
                            raise BackendError(
                                f"stream {resp.status_code} on {url}"
                            )
                        buffer: list[str] = []
                        async for line in resp.aiter_lines():
                            if line == "":
                                if buffer:
                                    raw = _parse_sse_block(buffer)
                                    buffer = []
                                    if raw is not None:
                                        evt = normalize(raw)
                                        seen_any_event = True
                                        yield evt
                                        if is_terminal(evt):
                                            return
                                continue
                            buffer.append(line)
                        # stream closed cleanly
                        return
            except (
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ) as exc:
                attempts += 1
                if attempts > max_reconnects:
                    raise BackendError(f"SSE reconnect limit reached: {exc}") from exc
                log.warning(
                    "SSE disconnect on %s (attempt %d/%d): %s",
                    url, attempts, max_reconnects, exc,
                )
                await asyncio.sleep(STREAM_RECONNECT_DELAY)


def _parse_sse_block(lines: list[str]) -> Optional[dict[str, Any]]:
    """Parse a single SSE event block (lines between blank lines)."""
    data_parts: list[str] = []
    for ln in lines:
        if ln.startswith(":"):
            continue  # comment / heartbeat
        if ln.startswith("data:"):
            data_parts.append(ln[5:].lstrip())
    if not data_parts:
        return None
    raw = "\n".join(data_parts)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


# ---------------------------------------------------------------------------
# Factory (DIP): build clients from env vars
# ---------------------------------------------------------------------------

_clients: dict[str, BackendClient] = {}


def _get(name: str, env_key: str, default: str) -> BackendClient:
    if name not in _clients:
        _clients[name] = BackendClient(os.environ.get(env_key, default))
    return _clients[name]


def splitter_client() -> BackendClient:
    return _get("splitter", "SPLITTER_URL", "http://localhost:8001")


def subs_client() -> BackendClient:
    return _get("subs", "SUBS_URL", "http://localhost:8002")


def dubbing_client() -> BackendClient:
    return _get("dubbing", "DUBBING_URL", "http://localhost:8003")


def reset_clients() -> None:
    """Test helper: clear the client cache so env vars get re-read."""
    _clients.clear()
