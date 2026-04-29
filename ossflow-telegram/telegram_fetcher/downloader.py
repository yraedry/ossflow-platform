"""Download a single Telegram message's media with progress + cancel.

Cancellation policy: on ``asyncio.CancelledError`` we delete the partial file
before re-raising, matching the decision in the planning doc.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Tuple

from .errors import MediaUnavailableError, TelegramError


log = logging.getLogger(__name__)


ProgressCB = Callable[[int, int], None]


class Downloader:
    """Minimal download wrapper around ``client.download_media``."""

    async def download(
        self,
        client: Any,
        channel_id: str,
        message_id: int,
        dest_path: str,
        progress_cb: Optional[ProgressCB] = None,
    ) -> Tuple[str, int]:
        """Download ``message_id`` from ``channel_id`` into ``dest_path``.

        Returns ``(final_path, bytes_downloaded)``. Deletes partial file on
        cancellation. Raises :class:`MediaUnavailableError` if the message has
        no media.
        """
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

        # Resolve the message so we can hand Telethon a concrete Message object.
        try:
            msg = await client.get_messages(channel_id, ids=int(message_id))
        except Exception as exc:  # noqa: BLE001
            raise TelegramError(f"get_messages failed: {exc}", cause=exc) from exc
        if msg is None or getattr(msg, "media", None) is None:
            raise MediaUnavailableError(
                f"message {message_id} has no media or was deleted"
            )

        wrapped_cb = progress_cb
        bytes_seen = {"n": 0}

        if progress_cb is not None:
            def _cb(current: int, total: int) -> None:
                bytes_seen["n"] = int(current)
                try:
                    progress_cb(int(current), int(total))
                except Exception:  # noqa: BLE001
                    log.exception("progress_cb raised")
            wrapped_cb = _cb

        try:
            saved = await client.download_media(
                msg, file=str(dest_path), progress_callback=wrapped_cb
            )
        except asyncio.CancelledError:
            self._cleanup_partial(dest_path)
            raise
        except MediaUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._cleanup_partial(dest_path)
            raise TelegramError(f"download_media failed: {exc}", cause=exc) from exc

        final_path = saved if isinstance(saved, str) else str(dest_path)
        try:
            size = os.path.getsize(final_path)
        except OSError:
            size = bytes_seen["n"]
        return final_path, int(size)

    @staticmethod
    def _cleanup_partial(path: str) -> None:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            log.warning("failed to remove partial %s", path)
