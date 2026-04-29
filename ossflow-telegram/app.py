"""FastAPI entrypoint for telegram-fetcher.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8004
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make bjj_service_kit importable when running from /app.
_KIT_PARENT = Path(__file__).resolve().parent.parent
if str(_KIT_PARENT) not in sys.path:
    sys.path.insert(0, str(_KIT_PARENT))

from bjj_service_kit import create_app, JobEvent, RunRequest  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from telegram_fetcher.auth_store import AuthStateStore  # noqa: E402
from telegram_fetcher.client import TelegramService  # noqa: E402
from telegram_fetcher.config import Config  # noqa: E402
from telegram_fetcher.db import Database  # noqa: E402
from telegram_fetcher.errors import (  # noqa: E402
    AuthFailedError,
    AuthRequiredError,
    ChannelNotFoundError,
    MediaUnavailableError,
    RateLimitError,
    TelegramError,
)
from telegram_fetcher.queue import (  # noqa: E402
    DownloadQueue,
    JobEventBus,
    SyncQueue,
    build_download_handler,
    build_sync_handler,
)
from telegram_fetcher.routers import (  # noqa: E402
    auth_router,
    channels_router,
    download_router,
    media_router,
)
from telegram_fetcher.routers.download import trigger_library_scan  # noqa: E402


SERVICE_NAME = "telegram-fetcher"
log = logging.getLogger(__name__)


def _noop_task(_req: RunRequest, _emit) -> None:  # pragma: no cover
    """Placeholder for the shared ``/run`` entrypoint (unused here)."""
    raise RuntimeError("telegram-fetcher does not expose a /run task")


def build_app() -> FastAPI:
    app = create_app(service_name=SERVICE_NAME, task_fn=_noop_task)

    # ------------------------------------------------------------------
    # State wiring (populated in lifespan)
    # ------------------------------------------------------------------
    app.state.db = None
    app.state.auth_store = None
    app.state.config = None
    app.state.telegram_service = None
    app.state.event_bus = None
    app.state.download_queue = None
    app.state.sync_queue = None
    app.state.download_completion_cb = None
    app.state._poll_task = None

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(auth_router)
    app.include_router(channels_router)
    app.include_router(media_router)
    app.include_router(download_router)

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------
    @app.exception_handler(TelegramError)
    async def _telegram_error_handler(_request: Request, exc: TelegramError):
        payload = exc.to_dict()
        headers = {}
        if isinstance(exc, RateLimitError) and exc.retry_after_s:
            headers["Retry-After"] = str(exc.retry_after_s)
        return JSONResponse(
            status_code=exc.http_status,
            content=payload,
            headers=headers or None,
        )

    # ------------------------------------------------------------------
    # Internal hot-reload
    # ------------------------------------------------------------------
    @app.post("/internal/reload-credentials")
    async def reload_credentials(request: Request) -> dict:
        config: Config = request.app.state.config
        if config is None:
            return {"ok": False, "reason": "config not ready"}
        changed = await config.reload()
        return {"ok": True, "changed": changed}

    # ------------------------------------------------------------------
    # Lifespan
    # ------------------------------------------------------------------
    @app.on_event("startup")
    async def _startup() -> None:
        await _do_startup(app)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await _do_shutdown(app)

    return app


async def _do_startup(app: FastAPI) -> None:
    # 1. Database
    db = Database()
    await db.init()
    orphaned = await db.reconcile_orphans()
    if orphaned:
        log.info("reconciled %d orphaned download jobs", orphaned)
    app.state.db = db

    # 2. Auth store
    auth_store = AuthStateStore.get_instance()
    app.state.auth_store = auth_store

    # 3. Event bus + queues
    bus = JobEventBus()
    app.state.event_bus = bus

    def _get_service():
        svc = app.state.telegram_service
        if svc is None:
            raise AuthRequiredError("Telegram service not configured (missing API credentials)")
        return svc

    download_handler = build_download_handler(_get_service, db)
    sync_handler = build_sync_handler(_get_service, db)
    dq = DownloadQueue(db, bus, download_handler)
    sq = SyncQueue(bus, sync_handler)
    await dq.start()
    await sq.start()
    app.state.download_queue = dq
    app.state.sync_queue = sq

    # 4. Config + credential polling
    config = Config()
    app.state.config = config

    async def _on_creds(api_id: int, api_hash: str) -> None:
        # Build (or refresh) the TelegramService.
        existing = app.state.telegram_service
        if existing is None:
            svc = TelegramService(api_id=api_id, api_hash=api_hash, auth_store=auth_store)
        else:
            svc = existing
            svc.set_credentials(api_id, api_hash)
        app.state.telegram_service = svc
        try:
            await svc.connect()
        except Exception:  # noqa: BLE001
            log.exception("post-credentials connect failed")

    config.on_change(_on_creds)
    app.state._poll_task = asyncio.create_task(config.poll_credentials_loop(), name="creds-poll")

    # 5. Download completion hook — triggers processor-api library scan.
    async def _on_download_done(job_id: str) -> None:
        q = await bus.subscribe(job_id)
        try:
            while True:
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=3600.0)
                except asyncio.TimeoutError:
                    return
                if evt.get("type") == "__end__":
                    return
                if evt.get("type") == "done":
                    await trigger_library_scan(config.processor_api_url)
                    return
        finally:
            await bus.unsubscribe(job_id, q)

    app.state.download_completion_cb = _on_download_done


async def _do_shutdown(app: FastAPI) -> None:
    # Stop poller
    poll_task = getattr(app.state, "_poll_task", None)
    config = getattr(app.state, "config", None)
    if config is not None:
        config.stop()
    if poll_task is not None:
        poll_task.cancel()
        try:
            await poll_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    # Stop queues
    dq = getattr(app.state, "download_queue", None)
    sq = getattr(app.state, "sync_queue", None)
    if dq is not None:
        await dq.stop()
    if sq is not None:
        await sq.stop()

    # Disconnect client
    svc = getattr(app.state, "telegram_service", None)
    if svc is not None:
        try:
            await svc.disconnect()
        except Exception:  # noqa: BLE001
            log.exception("telegram disconnect failed")

    # Close DB
    db = getattr(app.state, "db", None)
    if db is not None:
        await db.close()


app = build_app()
