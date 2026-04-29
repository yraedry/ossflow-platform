"""Tests for /telegram/media endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from telegram_fetcher.models import MediaItem


pytestmark = pytest.mark.asyncio


def _async_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _mk(mid: int, *, author="John Danaher", title="Leglocks", ch=1) -> MediaItem:
    return MediaItem(
        channel_id="C1",
        message_id=mid,
        caption=f"{author} - {title} - Capítulo {ch}",
        filename=f"f{mid}.mp4",
        size_bytes=1000 + mid,
        mime_type="video/mp4",
        date=datetime(2026, 4, 1, 12, mid, tzinfo=timezone.utc),
        author=author,
        title=title,
        chapter_num=ch,
    )


async def _seed(db) -> None:
    await db.upsert_media(_mk(1, ch=1))
    await db.upsert_media(_mk(2, ch=2))
    await db.upsert_media(_mk(3, author="Gordon Ryan", title="Guard Passing", ch=1))


async def test_media_chronological_shape_and_pagination(wired_app) -> None:
    app, db, *_ = wired_app
    await _seed(db)
    async with _async_client(app) as client:
        r = await client.get("/telegram/media", params={"view": "chronological", "page_size": 2, "page": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["view"] == "chronological"
    assert body["page"] == 1
    assert body["page_size"] == 2
    assert len(body["items"]) == 2
    first = body["items"][0]
    for key in ("channel_id", "message_id", "author", "title", "chapter_num"):
        assert key in first

    async with _async_client(app) as client:
        r2 = await client.get("/telegram/media", params={"view": "chronological", "page_size": 2, "page": 2})
    body2 = r2.json()
    assert len(body2["items"]) == 1
    # No overlap
    first_ids = {i["message_id"] for i in body["items"]}
    second_ids = {i["message_id"] for i in body2["items"]}
    assert first_ids.isdisjoint(second_ids)


async def test_media_by_author_groups(wired_app) -> None:
    app, db, *_ = wired_app
    await _seed(db)
    async with _async_client(app) as client:
        r = await client.get("/telegram/media", params={"view": "by_author"})
    assert r.status_code == 200
    body = r.json()
    assert body["view"] == "by_author"
    authors = {a["name"]: a for a in body["authors"]}
    assert "John Danaher" in authors
    assert "Gordon Ryan" in authors
    danaher = authors["John Danaher"]["instructionals"]
    assert len(danaher) == 1
    assert danaher[0]["title"] == "Leglocks"
    assert danaher[0]["chapters"] == 2
    ryan = authors["Gordon Ryan"]["instructionals"]
    assert ryan[0]["chapters"] == 1


async def test_update_metadata_marks_manual(wired_app) -> None:
    app, db, *_ = wired_app
    await _seed(db)
    async with _async_client(app) as client:
        r = await client.put(
            "/telegram/media/C1/1",
            json={"author": "Alt Author", "title": "Alt Title", "chapter_num": 9},
        )
    assert r.status_code == 200
    assert r.json() == {"ok": True, "manual_metadata": True}

    async with db.conn.execute(
        "SELECT author, title, chapter_num, manual_metadata FROM media WHERE channel_id='C1' AND message_id=1"
    ) as cur:
        row = await cur.fetchone()
    assert row["author"] == "Alt Author"
    assert row["title"] == "Alt Title"
    assert row["chapter_num"] == 9
    assert row["manual_metadata"] == 1
