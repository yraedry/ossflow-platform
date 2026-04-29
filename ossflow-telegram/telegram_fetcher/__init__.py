"""telegram-fetcher core package."""
from .errors import (
    AuthFailedError,
    AuthRequiredError,
    ChannelNotFoundError,
    MediaUnavailableError,
    RateLimitError,
    TelegramError,
)
from .models import (
    AuthState,
    AuthorView,
    DownloadJob,
    InstructionalGroup,
    MediaItem,
    SyncReport,
)
from .parser import ParsedCaption, parse_caption

__all__ = [
    "AuthFailedError",
    "AuthRequiredError",
    "AuthState",
    "AuthorView",
    "ChannelNotFoundError",
    "DownloadJob",
    "InstructionalGroup",
    "MediaItem",
    "MediaUnavailableError",
    "ParsedCaption",
    "RateLimitError",
    "SyncReport",
    "TelegramError",
    "parse_caption",
]
