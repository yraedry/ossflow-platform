"""ChannelBrowser: iterate channel messages and extract media metadata.

Consumes a Telethon ``TelegramClient`` (or a fake with the same surface) and
yields simple dicts that downstream code can turn into :class:`MediaItem`.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Optional

from .db import Database
from .errors import ChannelNotFoundError, TelegramError


log = logging.getLogger(__name__)


def _extract_media_dict(msg: Any) -> Optional[Dict[str, Any]]:
    """Return a plain dict describing the media attached to ``msg``.

    Returns ``None`` when the message carries no downloadable document.
    """
    media = getattr(msg, "media", None)
    if media is None:
        return None
    # Only MessageMediaDocument carries a Document (videos, files...).
    # We rely on telethon's File helper when available.
    doc = getattr(media, "document", None)
    if doc is None:
        return None

    filename: Optional[str] = None
    mime: Optional[str] = getattr(doc, "mime_type", None)
    size = int(getattr(doc, "size", 0) or 0)
    duration: Optional[float] = None

    attrs = getattr(doc, "attributes", None) or []
    for a in attrs:
        cn = type(a).__name__
        if cn == "DocumentAttributeFilename" and filename is None:
            filename = getattr(a, "file_name", None)
        elif cn == "DocumentAttributeVideo":
            duration = getattr(a, "duration", None)

    return {
        "message_id": int(getattr(msg, "id", 0)),
        "filename": filename,
        "mime_type": mime,
        "size_bytes": size,
        "duration": duration,
        "caption": getattr(msg, "message", None) or getattr(msg, "text", None),
        "date": getattr(msg, "date", None),
    }


class ChannelBrowser:
    """Iterate channel media and persist channel identity in the cache."""

    def __init__(self, db: Optional[Database] = None) -> None:
        self._db = db

    async def resolve_channel(self, client: Any, username: str) -> Dict[str, Any]:
        """Resolve a username/link to ``{channel_id, username, title}``.

        Persists the result in the ``channels`` table when a DB is configured.
        """
        try:
            entity = await client.get_entity(username)
        except Exception as exc:  # noqa: BLE001
            raise ChannelNotFoundError(f"cannot resolve channel {username!r}") from exc
        channel_id = str(getattr(entity, "id", ""))
        title = getattr(entity, "title", None)
        handle = getattr(entity, "username", None) or username.lstrip("@")
        noforwards = bool(getattr(entity, "noforwards", False))
        if self._db is not None:
            try:
                await self._db.upsert_channel(
                    channel_id, handle, title=title, noforwards=noforwards
                )
            except Exception:  # noqa: BLE001
                log.exception("failed to upsert channel %s", channel_id)
        return {
            "channel_id": channel_id,
            "username": handle,
            "title": title,
            "noforwards": noforwards,
        }

    async def iter_channel_media(
        self,
        client: Any,
        channel_username: str,
        *,
        limit: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield media dicts filtered to videos only."""
        # Deferred import so tests can mock client without Telethon installed.
        try:
            from telethon.tl.types import InputMessagesFilterVideo  # type: ignore
            video_filter: Any = InputMessagesFilterVideo
        except Exception:  # noqa: BLE001
            video_filter = None

        info = await self.resolve_channel(client, channel_username)
        channel_id = info["channel_id"]

        try:
            iterator = client.iter_messages(
                channel_username, limit=limit, filter=video_filter
            )
        except TypeError:
            # Fakes may not accept kwarg ``filter``; retry without it.
            iterator = client.iter_messages(channel_username, limit=limit)
        except Exception as exc:  # noqa: BLE001
            raise TelegramError(f"iter_messages failed: {exc}", cause=exc) from exc

        async for msg in iterator:
            data = _extract_media_dict(msg)
            if data is None:
                continue
            # Only keep video-like media (safety net when filter not honored).
            mime = data.get("mime_type") or ""
            if mime and not mime.startswith("video/"):
                continue
            data["channel_id"] = channel_id
            data["channel_username"] = info["username"]
            # Expose raw message so callers can download thumbnails without
            # re-fetching via get_messages.
            data["_msg"] = msg
            yield data
