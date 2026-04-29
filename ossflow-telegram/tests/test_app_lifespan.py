"""Tests for app startup/shutdown lifecycle."""
from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# Ensure bjj_service_kit on path.
_KIT_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_KIT_PARENT) not in sys.path:
    sys.path.insert(0, str(_KIT_PARENT))


pytestmark = pytest.mark.asyncio


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    cache_db = tmp_path / "cache" / "telegram.db"
    session = tmp_path / "session"
    session.mkdir()
    cache_db.parent.mkdir()
    monkeypatch.setenv("TG_CACHE_DB", str(cache_db))
    monkeypatch.setenv("TG_AUTH_META", str(session / "auth_meta.json"))
    monkeypatch.setenv("TG_SESSION_PATH", str(session / "session"))
    monkeypatch.setenv("TG_LIBRARY_ROOT", str(tmp_path / "library"))
    monkeypatch.setenv("PROCESSOR_API_URL", "http://fake:9999")

    # Reset auth singleton.
    from telegram_fetcher.auth_store import AuthStateStore
    AuthStateStore.reset_instance()
    yield tmp_path
    AuthStateStore.reset_instance()


@pytest.fixture
def patched_config(monkeypatch):
    """Prevent Config from hitting processor-api (fetch_once returns None)."""
    import telegram_fetcher.config as cfg_mod

    async def _fake_fetch(self):
        return None

    monkeypatch.setattr(cfg_mod.Config, "fetch_once", _fake_fetch)
    yield


async def test_startup_initializes_state_and_workers(isolated_env, patched_config):
    import app as app_module
    importlib.reload(app_module)
    fastapi_app = app_module.build_app()

    # Spy on key coroutines by wrapping.
    from telegram_fetcher.db import Database
    original_init = Database.init
    original_reconcile = Database.reconcile_orphans
    init_calls: list[int] = []
    reconcile_calls: list[int] = []

    async def _init(self):
        init_calls.append(1)
        return await original_init(self)

    async def _reconcile(self):
        reconcile_calls.append(1)
        return await original_reconcile(self)

    Database.init = _init  # type: ignore[method-assign]
    Database.reconcile_orphans = _reconcile  # type: ignore[method-assign]
    try:
        await app_module._do_startup(fastapi_app)

        assert fastapi_app.state.db is not None
        assert fastapi_app.state.auth_store is not None
        assert fastapi_app.state.event_bus is not None
        assert fastapi_app.state.download_queue is not None
        assert fastapi_app.state.sync_queue is not None
        assert fastapi_app.state.config is not None
        assert fastapi_app.state._poll_task is not None
        assert not fastapi_app.state._poll_task.done()
        # Download + sync workers running.
        assert fastapi_app.state.download_queue._worker is not None  # type: ignore[attr-defined]
        assert not fastapi_app.state.download_queue._worker.done()  # type: ignore[attr-defined]
        assert len(init_calls) == 1
        assert len(reconcile_calls) == 1
    finally:
        Database.init = original_init  # type: ignore[method-assign]
        Database.reconcile_orphans = original_reconcile  # type: ignore[method-assign]
        await app_module._do_shutdown(fastapi_app)


async def test_shutdown_cancels_workers_and_closes_db(isolated_env, patched_config):
    import app as app_module
    importlib.reload(app_module)
    fastapi_app = app_module.build_app()
    await app_module._do_startup(fastapi_app)

    # Install a fake telegram_service to verify disconnect() is called.
    fake_svc = MagicMock()
    fake_svc.disconnect = AsyncMock()
    fastapi_app.state.telegram_service = fake_svc

    dq = fastapi_app.state.download_queue
    sq = fastapi_app.state.sync_queue
    poll_task = fastapi_app.state._poll_task
    db = fastapi_app.state.db

    await app_module._do_shutdown(fastapi_app)

    assert dq._worker is None or dq._worker.done()  # type: ignore[attr-defined]
    assert sq._worker is None or sq._worker.done()  # type: ignore[attr-defined]
    assert poll_task.done()
    fake_svc.disconnect.assert_awaited_once()
    # DB closed: conn is None now.
    assert db._conn is None  # type: ignore[attr-defined]


async def test_startup_idempotent_double_call(isolated_env, patched_config):
    """Calling startup twice should not raise."""
    import app as app_module
    importlib.reload(app_module)
    fastapi_app = app_module.build_app()
    await app_module._do_startup(fastapi_app)
    # Second startup: don't crash, even though some resources already wired.
    try:
        await app_module._do_startup(fastapi_app)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"second startup raised: {exc!r}")
    finally:
        await app_module._do_shutdown(fastapi_app)
