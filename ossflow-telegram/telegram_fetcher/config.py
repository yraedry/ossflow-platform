"""Runtime configuration for telegram-fetcher.

Responsibilities:
- Expose the URL of processor-api (``PROCESSOR_API_URL`` env, default
  ``http://processor-api:8000``).
- Fetch ``telegram_api_id`` / ``telegram_api_hash`` from
  ``GET {PROCESSOR_API_URL}/api/settings`` with exponential backoff until
  they become available. While missing, the service stays in
  ``disconnected`` mode without raising.
- Allow hot-reload (see ``/internal/reload-credentials`` in ``app.py``).
- Notify a :class:`TelegramService` whenever credentials change.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Optional, Tuple

import httpx


log = logging.getLogger(__name__)


DEFAULT_PROCESSOR_API_URL = "http://processor-api:8000"
DEFAULT_POLL_MIN_S = 2.0
DEFAULT_POLL_MAX_S = 60.0
DEFAULT_POLL_TIMEOUT_S = 5.0


CredsChangedCB = Callable[[int, str], Awaitable[None]]


class Config:
    """Credentials poller.

    Instances are safe to reuse; callers typically keep one alive on
    ``app.state.config`` during the FastAPI lifespan.
    """

    def __init__(
        self,
        processor_api_url: Optional[str] = None,
        *,
        min_backoff_s: float = DEFAULT_POLL_MIN_S,
        max_backoff_s: float = DEFAULT_POLL_MAX_S,
        http_timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
        http_client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
    ) -> None:
        self.processor_api_url = (
            processor_api_url
            or os.environ.get("PROCESSOR_API_URL")
            or DEFAULT_PROCESSOR_API_URL
        ).rstrip("/")
        self.min_backoff_s = float(min_backoff_s)
        self.max_backoff_s = float(max_backoff_s)
        self.http_timeout_s = float(http_timeout_s)
        self._http_client_factory = http_client_factory or (
            lambda: httpx.AsyncClient(timeout=self.http_timeout_s)
        )
        self._api_id: Optional[int] = None
        self._api_hash: Optional[str] = None
        self._listeners: list[CredsChangedCB] = []
        self._stop_event = asyncio.Event()
        self._have_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def api_id(self) -> Optional[int]:
        return self._api_id

    @property
    def api_hash(self) -> Optional[str]:
        return self._api_hash

    def have_credentials(self) -> bool:
        return self._api_id is not None and self._api_hash is not None

    def on_change(self, cb: CredsChangedCB) -> None:
        """Register a callback invoked when credentials first arrive or change."""
        self._listeners.append(cb)

    def stop(self) -> None:
        """Signal the polling loop to exit at the next iteration."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Fetch logic
    # ------------------------------------------------------------------

    async def fetch_once(self) -> Optional[Tuple[int, str]]:
        """Single HTTP call. Returns ``(api_id, api_hash)`` or ``None``.

        Never raises on transport or HTTP error — logs and returns ``None``.
        """
        # /internal returns secrets unmasked (network-restricted to the
        # private Docker subnet). The public /api/settings replaces
        # telegram_api_hash with "***" for the frontend, which would make
        # Telethon raise ApiIdInvalidError.
        url = f"{self.processor_api_url}/api/settings/internal"
        try:
            async with self._http_client_factory() as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001
            log.info("settings fetch failed (%s): %s", url, exc)
            return None

        api_id_raw = data.get("telegram_api_id")
        api_hash = data.get("telegram_api_hash")
        if not api_id_raw or not api_hash:
            return None
        try:
            api_id = int(api_id_raw)
        except (TypeError, ValueError):
            log.warning("invalid telegram_api_id in settings: %r", api_id_raw)
            return None
        if not isinstance(api_hash, str) or not api_hash:
            return None
        return api_id, api_hash

    async def _apply(self, api_id: int, api_hash: str) -> bool:
        """Store and notify listeners. Returns ``True`` if it differs from cache."""
        if self._api_id == api_id and self._api_hash == api_hash:
            return False
        self._api_id = api_id
        self._api_hash = api_hash
        self._have_event.set()
        for cb in list(self._listeners):
            try:
                await cb(api_id, api_hash)
            except Exception:  # noqa: BLE001
                log.exception("credential listener raised")
        return True

    async def reload(self) -> bool:
        """Hot-reload entrypoint. Returns True if credentials changed."""
        res = await self.fetch_once()
        if res is None:
            return False
        return await self._apply(*res)

    async def wait_for_credentials(self, timeout: Optional[float] = None) -> bool:
        """Block until credentials have been fetched at least once."""
        try:
            await asyncio.wait_for(self._have_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def poll_credentials_loop(self) -> None:
        """Background task: poll settings with exponential backoff.

        The loop never exits on its own — call :meth:`stop` to break it.
        """
        delay = self.min_backoff_s
        while not self._stop_event.is_set():
            res = await self.fetch_once()
            if res is not None:
                await self._apply(*res)
                delay = self.max_backoff_s  # back off once we have creds
            else:
                delay = min(delay * 2.0, self.max_backoff_s) if delay else self.min_backoff_s

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue
