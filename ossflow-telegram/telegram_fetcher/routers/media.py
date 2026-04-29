"""Media listing + manual metadata editing."""
from __future__ import annotations

from collections import defaultdict
from typing import Literal, Optional

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel


router = APIRouter(prefix="/telegram", tags=["telegram-media"])


def _thumb_url(channel_id: Optional[str], message_id: Optional[int]) -> Optional[str]:
    if not channel_id or message_id is None:
        return None
    return f"/telegram/media/{channel_id}/{message_id}/thumbnail"


class _MetadataBody(BaseModel):
    author: Optional[str] = None
    title: Optional[str] = None
    chapter_num: Optional[int] = None


@router.get("/media")
async def list_media(
    request: Request,
    channel: Optional[str] = Query(default=None),
    view: Literal["chronological", "by_author"] = Query(default="chronological"),
    search: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
) -> dict:
    db = request.app.state.db
    channel_id = await _resolve_channel_id(db, channel)

    if view == "by_author":
        groups = await db.group_by_author(channel_id=channel_id)
        # Bucketize by author.
        buckets: dict[str, list[dict]] = defaultdict(list)
        for g in groups:
            thumb_url = _thumb_url(g.first_channel_id, g.first_message_id) \
                if g.first_thumbnail_path else None
            buckets[g.author].append({
                "title": g.title,
                "chapters": g.chapter_count,
                "total_size_bytes": g.total_size_bytes,
                "available": g.available,
                "downloaded_chapters": g.downloaded_chapters,
                "message_ids": g.message_ids,
                "first_channel_id": g.first_channel_id,
                "first_message_id": g.first_message_id,
                "thumbnail_url": thumb_url,
            })
        authors = [
            {"name": name, "instructionals": items}
            for name, items in sorted(buckets.items())
        ]
        return {"view": "by_author", "authors": authors}

    offset = (page - 1) * page_size
    items = await db.list_media(
        channel_id=channel_id, limit=page_size, offset=offset, search=search
    )
    serialized = []
    for m in items:
        row = m.model_dump(mode="json")
        if m.thumbnail_path:
            row["thumbnail_url"] = _thumb_url(m.channel_id, m.message_id)
        else:
            row["thumbnail_url"] = None
        serialized.append(row)
    return {
        "view": "chronological",
        "page": page,
        "page_size": page_size,
        "items": serialized,
    }


@router.get("/media/{channel_id}/{message_id}/thumbnail")
async def get_thumbnail(channel_id: str, message_id: int, request: Request):
    db = request.app.state.db
    path = await db.get_thumbnail_path(channel_id, int(message_id))
    if not path:
        raise HTTPException(status_code=404, detail="thumbnail not cached")
    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="thumbnail file missing")
    return FileResponse(
        str(p),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.put("/media/{channel_id}/{message_id}")
async def update_metadata(
    channel_id: str,
    message_id: int,
    body: _MetadataBody,
    request: Request,
) -> dict:
    db = request.app.state.db
    await db.update_metadata(
        channel_id,
        int(message_id),
        author=body.author,
        title=body.title,
        chapter_num=body.chapter_num,
        manual=True,
    )
    return {"ok": True, "manual_metadata": True}


async def _resolve_channel_id(db, channel: Optional[str]) -> Optional[str]:
    """Accept either a numeric channel_id or a username. Returns channel_id."""
    if not channel:
        return None
    if channel.lstrip("-").isdigit():
        return channel
    async with db.conn.execute(
        "SELECT channel_id FROM channels WHERE username = ?", (channel.lstrip("@"),)
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        # Allow filtering by unknown username — returns empty results rather than 404.
        return channel
    return row["channel_id"]
