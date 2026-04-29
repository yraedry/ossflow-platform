"""Shared fixtures for router + lifespan tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# Make bjj_service_kit importable.
_KIT_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_KIT_PARENT) not in sys.path:
    sys.path.insert(0, str(_KIT_PARENT))


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    """Point all on-disk paths at tmp_path before importing app/db/auth_store."""
    cache_db = tmp_path / "cache" / "telegram.db"
    session_dir = tmp_path / "session"
    auth_meta = session_dir / "auth_meta.json"
    library = tmp_path / "library"
    library.mkdir()
    session_dir.mkdir(parents=True, exist_ok=True)
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TG_CACHE_DB", str(cache_db))
    monkeypatch.setenv("TG_AUTH_META", str(auth_meta))
    monkeypatch.setenv("TG_SESSION_PATH", str(session_dir / "session"))
    monkeypatch.setenv("TG_LIBRARY_ROOT", str(library))
    monkeypatch.setenv("PROCESSOR_API_URL", "http://fake-processor:9999")
    # Reset auth singleton.
    from telegram_fetcher.auth_store import AuthStateStore
    AuthStateStore.reset_instance()
    yield {"tmp_path": tmp_path, "library": library, "auth_meta": auth_meta}
    AuthStateStore.reset_instance()


class FakeTelegramService:
    """Mimics TelegramService for router tests."""

    def __init__(self, auth_store) -> None:
        self._auth = auth_store
        self.connected = True
        self.send_code = AsyncMock(side_effect=self._send_code)
        self.sign_in_code = AsyncMock(side_effect=self._sign_in_code)
        self.sign_in_2fa = AsyncMock(side_effect=self._sign_in_2fa)
        self.logout = AsyncMock(side_effect=self._logout)
        self.disconnect = AsyncMock()
        self.next_code_behavior = "ok"  # "ok", "needs_2fa", "auth_failed"
        self.next_send_behavior = "ok"  # "ok", "auth_failed"
        self.next_2fa_behavior = "ok"
        self.client = MagicMock()

    async def _is_connected(self) -> bool:
        return self.connected

    async def _send_code(self, phone: str) -> str:
        if self.next_send_behavior == "auth_failed":
            from telegram_fetcher.errors import AuthFailedError
            raise AuthFailedError("PhoneNumberInvalidError")
        self._auth.set_awaiting_code(phone, "hash-xyz")
        return "hash-xyz"

    async def _sign_in_code(self, phone: str, code: str, phone_code_hash: str) -> None:
        if self.next_code_behavior == "auth_failed":
            from telegram_fetcher.errors import AuthFailedError
            raise AuthFailedError("PhoneCodeInvalidError")
        if self.next_code_behavior == "needs_2fa":
            self._auth.set_awaiting_2fa()
            return
        self._auth.set_authenticated("tester")

    async def _sign_in_2fa(self, password: str) -> None:
        if self.next_2fa_behavior == "auth_failed":
            from telegram_fetcher.errors import AuthFailedError
            raise AuthFailedError("PasswordHashInvalidError")
        self._auth.set_authenticated("tester")

    async def _logout(self) -> None:
        self._auth.set_disconnected()


@pytest_asyncio.fixture
async def wired_app(tmp_env):
    """Build the FastAPI app with DB + auth_store + queues wired (no real lifespan)."""
    # Fresh imports to guarantee a new `app` instance per test.
    import importlib
    import telegram_fetcher.auth_store as _as
    importlib.reload(_as)
    from telegram_fetcher.auth_store import AuthStateStore
    from telegram_fetcher.db import Database
    from telegram_fetcher.queue import (
        DownloadQueue, SyncQueue, JobEventBus,
        build_download_handler, build_sync_handler,
    )
    from telegram_fetcher.config import Config

    # Build app using the same factory as production, but skip the startup
    # event (we wire state manually).
    import app as app_module
    importlib.reload(app_module)
    fastapi_app = app_module.build_app()
    # Strip startup/shutdown handlers so TestClient doesn't trigger real polling.
    fastapi_app.router.on_startup.clear()
    fastapi_app.router.on_shutdown.clear()

    db = Database()
    await db.init()
    auth_store = AuthStateStore.get_instance()
    bus = JobEventBus()

    def _svc_provider():
        svc = fastapi_app.state.telegram_service
        if svc is None:
            from telegram_fetcher.errors import AuthRequiredError
            raise AuthRequiredError("no service")
        return svc

    dq = DownloadQueue(db, bus, build_download_handler(_svc_provider, db))
    sq = SyncQueue(bus, build_sync_handler(_svc_provider, db))
    await dq.start()
    await sq.start()

    config = Config()
    fake_svc = FakeTelegramService(auth_store)

    fastapi_app.state.db = db
    fastapi_app.state.auth_store = auth_store
    fastapi_app.state.event_bus = bus
    fastapi_app.state.download_queue = dq
    fastapi_app.state.sync_queue = sq
    fastapi_app.state.config = config
    fastapi_app.state.telegram_service = fake_svc
    fastapi_app.state.download_completion_cb = None

    try:
        yield fastapi_app, db, fake_svc, auth_store
    finally:
        # Cancel any lingering in-flight handler task before stopping workers.
        for q in (dq, sq):
            t = getattr(q, "_current_task", None)
            if t is not None and not t.done():
                t.cancel()
                try:
                    await t
                except BaseException:  # noqa: BLE001
                    pass
        await dq.stop()
        await sq.stop()
        await db.close()
