"""Telethon wrapper.

Owns a single :class:`telethon.TelegramClient` instance pointing at a session
file inside ``/data/session`` (overridable via env). Exposes the minimum set of
auth operations needed by the HTTP layer (T3). All Telethon errors are
translated into :mod:`telegram_fetcher.errors` domain errors.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .auth_store import AuthStateStore
from .errors import (
    AuthFailedError,
    AuthRequiredError,
    RateLimitError,
    TelegramError,
)


log = logging.getLogger(__name__)


DEFAULT_SESSION_PATH = "/data/session/session"  # Telethon appends .session (SQLite)
FLOOD_THRESHOLD_S = 60  # Re-raise as RateLimitError when wait > threshold


def _resolve_session_path(path: Optional[str]) -> str:
    raw = path or os.environ.get("TG_SESSION_PATH") or DEFAULT_SESSION_PATH
    # Strip trailing .session if passed in.
    if raw.endswith(".session"):
        raw = raw[: -len(".session")]
    Path(raw).parent.mkdir(parents=True, exist_ok=True)
    return raw


def ensure_connected(method: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorator: transparently reconnect before calling the wrapped method.

    Raises :class:`AuthRequiredError` if the session is not authorized.
    """

    @functools.wraps(method)
    async def wrapper(self: "TelegramService", *args: Any, **kwargs: Any) -> Any:
        await self._ensure_connected()
        return await method(self, *args, **kwargs)

    return wrapper


class TelegramService:
    """High-level facade around :class:`telethon.TelegramClient`."""

    def __init__(
        self,
        api_id: Optional[int] = None,
        api_hash: Optional[str] = None,
        *,
        session_path: Optional[str] = None,
        auth_store: Optional[AuthStateStore] = None,
    ) -> None:
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_path = _resolve_session_path(session_path)
        self._client: Any = None  # telethon.TelegramClient (lazy import)
        self._auth = auth_store or AuthStateStore.get_instance()
        self._client_factory: Optional[Callable[..., Any]] = None  # test hook

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_credentials(self, api_id: int, api_hash: str) -> None:
        self.api_id = int(api_id)
        self.api_hash = api_hash

    def set_client_factory(self, factory: Callable[..., Any]) -> None:
        """Inject a client factory (used by tests to supply a fake)."""
        self._client_factory = factory
        self._client = None

    def _build_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(self.session_path, self.api_id, self.api_hash)
        # Deferred import so tests / static checks don't need telethon installed.
        from telethon import TelegramClient  # type: ignore
        return TelegramClient(
            self.session_path,
            self.api_id,
            self.api_hash,
            flood_sleep_threshold=FLOOD_THRESHOLD_S,
        )

    async def connect(self) -> None:
        if self.api_id is None or self.api_hash is None:
            raise AuthRequiredError("missing API credentials")
        if self._client is None:
            self._client = self._build_client()
        try:
            if not await self._is_connected():
                await self._client.connect()
        except Exception as exc:  # noqa: BLE001
            raise TelegramError(f"connect failed: {exc}", cause=exc) from exc

        if await self._client.is_user_authorized():
            me = None
            try:
                me = await self._client.get_me()
            except Exception:  # noqa: BLE001
                me = None
            username = getattr(me, "username", None) if me else None
            # Only transition if not already authenticated (avoid spurious state churn).
            if self._auth.get_state().state != "authenticated":
                # Force direct set via disconnected->authenticated shortcut.
                try:
                    self._auth.set_authenticated(username)
                except Exception:  # noqa: BLE001
                    pass

    async def disconnect(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.disconnect()
        except Exception:  # noqa: BLE001
            pass

    async def _is_connected(self) -> bool:
        if self._client is None:
            return False
        try:
            res = self._client.is_connected()
            if asyncio.iscoroutine(res):
                return bool(await res)
            return bool(res)
        except Exception:  # noqa: BLE001
            return False

    async def _ensure_connected(self) -> None:
        if self.api_id is None or self.api_hash is None:
            raise AuthRequiredError("missing API credentials")
        if self._client is None or not await self._is_connected():
            await self.connect()
        if not await self._client.is_user_authorized():
            raise AuthRequiredError("session is not authorized")

    # ------------------------------------------------------------------
    # Auth flow
    # ------------------------------------------------------------------

    async def send_code(self, phone: str) -> str:
        """Request Telegram to send an SMS/app code. Returns ``phone_code_hash``."""
        await self.connect()
        try:
            sent = await self._client.send_code_request(phone)
        except Exception as exc:
            self._translate_and_raise(exc)
        phone_code_hash = getattr(sent, "phone_code_hash", None) or ""
        self._auth.set_awaiting_code(phone, phone_code_hash)
        return phone_code_hash

    async def sign_in_code(self, phone: str, code: str, phone_code_hash: str) -> None:
        await self.connect()
        try:
            await self._client.sign_in(phone=phone, code=code, phone_code_hash=phone_code_hash)
        except Exception as exc:
            # Lazy import: Telethon may not be present in test env.
            if _is_password_needed(exc):
                self._auth.set_awaiting_2fa()
                return
            self._translate_and_raise(exc)
        me = None
        try:
            me = await self._client.get_me()
        except Exception:  # noqa: BLE001
            pass
        self._auth.set_authenticated(getattr(me, "username", None) if me else None)

    async def sign_in_2fa(self, password: str) -> None:
        await self.connect()
        try:
            await self._client.sign_in(password=password)
        except Exception as exc:
            self._translate_and_raise(exc)
        me = None
        try:
            me = await self._client.get_me()
        except Exception:  # noqa: BLE001
            pass
        self._auth.set_authenticated(getattr(me, "username", None) if me else None)

    async def logout(self) -> None:
        try:
            if self._client is not None:
                await self._client.log_out()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._auth.set_disconnected()
            await self.disconnect()

    # ------------------------------------------------------------------
    # Accessors for T3 (browser / downloader)
    # ------------------------------------------------------------------

    async def ensure_ready(self) -> Any:
        """Ensure client is connected + authorized, then return it."""
        await self._ensure_connected()
        return self._client

    @property
    def client(self) -> Any:
        if self._client is None:
            raise AuthRequiredError("TelegramService not connected")
        return self._client

    # ------------------------------------------------------------------
    # Error translation
    # ------------------------------------------------------------------

    def _translate_and_raise(self, exc: BaseException) -> None:
        """Translate a Telethon exception into a domain error and raise.

        FloodWaitError: if ``seconds > FLOOD_THRESHOLD_S`` → RateLimitError;
        else sleep inline then re-raise (caller retries).
        """
        name = type(exc).__name__
        seconds = getattr(exc, "seconds", 0) or 0
        if name == "FloodWaitError":
            if seconds > FLOOD_THRESHOLD_S:
                raise RateLimitError(
                    f"FloodWait {seconds}s", retry_after_s=int(seconds), cause=exc
                ) from exc
            # Under threshold: Telethon usually auto-sleeps already; treat as rate limit small.
            raise RateLimitError(
                f"FloodWait {seconds}s", retry_after_s=int(seconds), cause=exc
            ) from exc
        if name in {"AuthKeyError", "AuthKeyUnregisteredError", "SessionRevokedError",
                    "AuthKeyDuplicatedError", "UserDeactivatedError"}:
            self._auth.set_disconnected()
            raise AuthRequiredError(f"session invalid: {name}", cause=exc) from exc
        if name in {"PhoneCodeInvalidError", "PhoneCodeExpiredError",
                    "PasswordHashInvalidError", "PhoneNumberInvalidError"}:
            raise AuthFailedError(f"{name}", cause=exc) from exc
        if name == "SessionPasswordNeededError":
            # Shouldn't reach here (caller handles it), but map just in case.
            self._auth.set_awaiting_2fa()
            raise AuthFailedError("2FA required", cause=exc) from exc
        # Unknown: wrap as generic domain error to keep layer boundaries clean.
        raise TelegramError(f"{name}: {exc}", cause=exc) from exc


def _is_password_needed(exc: BaseException) -> bool:
    return type(exc).__name__ == "SessionPasswordNeededError"
