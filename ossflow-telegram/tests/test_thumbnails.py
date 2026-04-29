"""Tests for thumbnail persistence + endpoint (schema v2)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from telegram_fetcher.db import Database
from telegram_fetcher.models import MediaItem


pytestmark = pytest.mark.asyncio


def _mk_media(mid: int = 1) -> MediaItem:
    return MediaItem(
        channel_id="C1",
        message_id=mid,
        caption="x",
        filename=f"f{mid}.mp4",
        size_bytes=10,
        mime_type="video/mp4",
        date=datetime(2026, 4, 13, tzinfo=timezone.utc),
        author="Gordon",
        title="Leglocks",
        chapter_num=1,
    )


class TestThumbnailColumn:
    async def test_column_added_and_roundtrip(self, tmp_path: Path) -> None:
        db = Database(str(tmp_path / "t.db"))
        await db.init()
        try:
            await db.upsert_media(_mk_media(1))
            assert await db.get_thumbnail_path("C1", 1) is None
            await db.set_thumbnail_path("C1", 1, "/tmp/x.jpg")
            assert await db.get_thumbnail_path("C1", 1) == "/tmp/x.jpg"
            # list_media round-trips thumbnail_path
            items = await db.list_media("C1")
            assert items[0].thumbnail_path == "/tmp/x.jpg"
        finally:
            await db.close()

    async def test_init_twice_idempotent(self, tmp_path: Path) -> None:
        # Second init should NOT raise "duplicate column name".
        p = tmp_path / "t.db"
        db = Database(str(p))
        await db.init()
        await db.close()
        db2 = Database(str(p))
        await db2.init()
        await db2.close()

    async def test_group_by_author_exposes_first_thumb(self, tmp_path: Path) -> None:
        db = Database(str(tmp_path / "t.db"))
        await db.init()
        try:
            await db.upsert_media(_mk_media(10))  # chapter_num=1
            m2 = _mk_media(11)
            m2.chapter_num = 2
            await db.upsert_media(m2)
            await db.set_thumbnail_path("C1", 10, "/tmp/a.jpg")
            await db.set_thumbnail_path("C1", 11, "/tmp/b.jpg")
            groups = await db.group_by_author("C1")
            assert len(groups) == 1
            g = groups[0]
            assert g.first_message_id == 10  # chapter 1 wins
            assert g.first_thumbnail_path == "/tmp/a.jpg"
            assert g.first_channel_id == "C1"
        finally:
            await db.close()
