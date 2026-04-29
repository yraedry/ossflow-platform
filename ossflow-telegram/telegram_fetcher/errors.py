"""Error hierarchy for telegram-fetcher.

Mapped to HTTP status codes by the (future) FastAPI layer:
    TelegramError           -> 500 (base / unknown)
    AuthRequiredError       -> 401
    AuthFailedError         -> 403
    RateLimitError          -> 429  (.retry_after_s)
    ChannelNotFoundError    -> 404
    MediaUnavailableError   -> 410
"""
from __future__ import annotations

from typing import Optional


class TelegramError(Exception):
    """Base error for all telegram-fetcher domain errors."""

    http_status: int = 500

    def __init__(self, message: str = "", *, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def to_dict(self) -> dict:
        return {"code": self.__class__.__name__, "message": self.message}


class AuthRequiredError(TelegramError):
    """No active Telegram session; UI must run auth flow."""

    http_status = 401


class AuthFailedError(TelegramError):
    """Wrong code / 2FA password / invalid phone."""

    http_status = 403


class RateLimitError(TelegramError):
    """Telethon FloodWaitError exceeding the auto-sleep threshold."""

    http_status = 429

    def __init__(self, message: str = "", *, retry_after_s: int = 0,
                 cause: Optional[BaseException] = None) -> None:
        super().__init__(message, cause=cause)
        self.retry_after_s = int(retry_after_s)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["retry_after"] = self.retry_after_s
        return d


class ChannelNotFoundError(TelegramError):
    """Channel username/id cannot be resolved."""

    http_status = 404


class MediaUnavailableError(TelegramError):
    """Message or media is no longer available (deleted/expired)."""

    http_status = 410
