"""Video message thumbnail download + filesystem layout.

Telethon exposes tiny preview thumbnails on ``msg.file.thumbs``. We fetch the
smallest thumbnail once per ``(channel_id, message_id)`` and cache it on disk
at ``{cache_dir}/{channel_id}_{message_id}.jpg``. Failures never abort the
sync — callers should treat exceptions as "no thumbnail this time".
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional


log = logging.getLogger(__name__)


DEFAULT_THUMB_DIR_ENV = "TG_THUMB_DIR"
DEFAULT_THUMB_DIR = "/data/cache/thumbs"


def thumb_dir() -> Path:
    return Path(os.environ.get(DEFAULT_THUMB_DIR_ENV) or DEFAULT_THUMB_DIR)


def thumb_path_for(channel_id: str, message_id: int) -> Path:
    safe_channel = str(channel_id).replace("/", "_").replace("\\", "_")
    return thumb_dir() / f"{safe_channel}_{int(message_id)}.jpg"


async def ensure_thumbnail(
    client: Any,
    msg: Any,
    channel_id: str,
    message_id: int,
) -> Optional[str]:
    """Download the smallest thumbnail for ``msg`` if missing.

    Returns the on-disk path as a string, or ``None`` if the message has no
    thumbnails or the download failed (logged at WARNING).
    """
    dest = thumb_path_for(channel_id, message_id)
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    # Guard: not all messages have thumbnails.
    thumbs = None
    try:
        f = getattr(msg, "file", None)
        if f is not None:
            thumbs = getattr(f, "thumbs", None)
    except Exception:  # noqa: BLE001
        thumbs = None
    if not thumbs:
        return None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        saved = await client.download_media(msg, file=str(dest), thumb=-1)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "thumbnail download failed for %s/%s: %s",
            channel_id,
            message_id,
            exc,
        )
        # Clean up empty stub if Telethon created one.
        try:
            if dest.exists() and dest.stat().st_size == 0:
                dest.unlink()
        except OSError:
            pass
        return None
    if saved is None:
        return None
    return str(saved) if isinstance(saved, (str, os.PathLike)) else str(dest)
