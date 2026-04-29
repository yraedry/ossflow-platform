"""Tests for Database wrapper."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from telegram_fetcher.db import (
    CURRENT_SCHEMA_VERSION,
    Database,
    _canonical_author_key,
    _canonical_title_key,
)
from telegram_fetcher.models import DownloadJob, MediaItem


pytestmark = pytest.mark.asyncio


def _mk_media(mid: int, *, channel: str = "C1", author: str | None = "John",
              title: str | None = "Leglocks", ch: int | None = 1) -> MediaItem:
    return MediaItem(
        channel_id=channel,
        message_id=mid,
        caption=f"cap {mid}",
        filename=f"file{mid}.mp4",
        size_bytes=100 + mid,
        mime_type="video/mp4",
        date=datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc),
        author=author,
        title=title,
        chapter_num=ch,
    )


async def _fresh_db(tmp_path: Path) -> Database:
    db = Database(str(tmp_path / "t.db"))
    await db.init()
    return db


class TestSchema:
    async def test_init_creates_tables_and_version(self, tmp_path: Path) -> None:
        db = await _fresh_db(tmp_path)
        try:
            async with db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cur:
                names = {row[0] for row in await cur.fetchall()}
            assert {"media", "download_jobs", "channels", "schema_version"} <= names
            async with db.conn.execute("SELECT MAX(version) FROM schema_version") as cur:
                row = await cur.fetchone()
            assert row[0] == CURRENT_SCHEMA_VERSION
        finally:
            await db.close()

    async def test_init_is_idempotent(self, tmp_path: Path) -> None:
        db = await _fresh_db(tmp_path)
        await db.close()
        db2 = Database(str(tmp_path / "t.db"))
        await db2.init()  # must not raise
        await db2.close()


class TestMediaCRUD:
    async def test_upsert_and_list(self, tmp_path: Path) -> None:
        db = await _fresh_db(tmp_path)
        try:
            await db.upsert_media(_mk_media(1))
            await db.upsert_media(_mk_media(2, ch=2))
            rows = await db.list_media(channel_id="C1")
            assert len(rows) == 2
            assert {r.message_id for r in rows} == {1, 2}
        finally:
            await db.close()

    async def test_upsert_preserves_manual_metadata(self, tmp_path: Path) -> None:
        db = await _fresh_db(tmp_path)
        try:
            await db.upsert_media(_mk_media(1))
            await db.update_metadata("C1", 1, author="Manual", title="T", chapter_num=9, manual=True)
            # Simulate a rescan pushing different parsed values — must not overwrite.
            auto = _mk_media(1)
            auto.author = "AutoOther"
            auto.title = "OtherTitle"
            auto.chapter_num = 5
            await db.upsert_media(auto)
            rows = await db.list_media(channel_id="C1")
            assert rows[0].author == "Manual"
            assert rows[0].title == "T"
            assert rows[0].chapter_num == 9
        finally:
            await db.close()

    async def test_group_by_author(self, tmp_path: Path) -> None:
        db = await _fresh_db(tmp_path)
        try:
            await db.upsert_media(_mk_media(1, author="John", title="Leglocks", ch=1))
            await db.upsert_media(_mk_media(2, author="John", title="Leglocks", ch=2))
            await db.upsert_media(_mk_media(3, author="John", title="Leglocks", ch=3))
            await db.upsert_media(_mk_media(4, author="Craig", title="Passing", ch=1))
            await db.upsert_media(_mk_media(5, author=None, title=None, ch=None))
            groups = await db.group_by_author()
            # Null author/title row is excluded.
            assert len(groups) == 2
            leglocks = next(g for g in groups if g.title == "Leglocks")
            assert leglocks.chapter_count == 3
            assert leglocks.message_ids == [1, 2, 3]
            assert leglocks.available is True
        finally:
            await db.close()

    async def test_search_filter(self, tmp_path: Path) -> None:
        db = await _fresh_db(tmp_path)
        try:
            await db.upsert_media(_mk_media(1, title="Leglocks"))
            await db.upsert_media(_mk_media(2, title="Armlocks"))
            rows = await db.list_media(search="Arm")
            assert len(rows) == 1
            assert rows[0].title == "Armlocks"
        finally:
            await db.close()


class TestCanonicalKeys:
    def test_author_key_strips_trailing_vol(self) -> None:
        assert _canonical_author_key("Gordon Ryan") == _canonical_author_key("Gordon Ryan Vol")

    def test_author_key_strips_volume_n(self) -> None:
        assert _canonical_author_key("Gordon Ryan") == _canonical_author_key("Gordon Ryan Volume 2")

    def test_author_key_case_and_accents(self) -> None:
        assert _canonical_author_key("JOAO MIYAO") == _canonical_author_key("João Miyão")

    def test_author_key_ignores_punctuation(self) -> None:
        assert _canonical_author_key("Gordon, Ryan.") == _canonical_author_key("Gordon Ryan")

    def test_title_key_strips_trailing_number(self) -> None:
        assert _canonical_title_key("Back Attacks 2") == _canonical_title_key("Back Attacks")

    def test_title_key_strips_volume_parens(self) -> None:
        assert _canonical_title_key("Back Attacks (Vol 1)") == _canonical_title_key("Back Attacks")


class TestGroupByAuthorDedup:
    async def test_groups_gordon_ryan_and_gordon_ryan_vol(self, tmp_path: Path) -> None:
        db = await _fresh_db(tmp_path)
        try:
            await db.upsert_media(_mk_media(1, author="Gordon Ryan", title="Systematically Attacking The Guard", ch=1))
            await db.upsert_media(_mk_media(2, author="Gordon Ryan Vol", title="Systematically Attacking The Guard", ch=2))
            await db.upsert_media(_mk_media(3, author="Gordon Ryan", title="Systematically Attacking The Guard", ch=3))
            groups = await db.group_by_author()
            assert len(groups) == 1
            g = groups[0]
            # Display name should be the shorter (cleaner) variant.
            assert g.author == "Gordon Ryan"
            assert g.chapter_count == 3
            assert g.message_ids == [1, 2, 3]
        finally:
            await db.close()

    async def test_groups_title_variants_with_vol_residue(self, tmp_path: Path) -> None:
        db = await _fresh_db(tmp_path)
        try:
            await db.upsert_media(_mk_media(1, author="John Danaher", title="Back Attacks", ch=1))
            await db.upsert_media(_mk_media(2, author="John Danaher", title="Back Attacks 2", ch=2))
            groups = await db.group_by_author()
            assert len(groups) == 1
            assert groups[0].chapter_count == 2
        finally:
            await db.close()

    async def test_empty_canonical_author_excluded(self, tmp_path: Path) -> None:
        db = await _fresh_db(tmp_path)
        try:
            # author="Vol" normalizes to empty canonical key → dropped.
            await db.upsert_media(_mk_media(1, author="Vol", title="Whatever", ch=1))
            await db.upsert_media(_mk_media(2, author="Real Name", title="Whatever", ch=1))
            groups = await db.group_by_author()
            assert len(groups) == 1
            assert groups[0].author == "Real Name"
        finally:
            await db.close()


class TestDownloadJobs:
    async def test_enqueue_next_mark(self, tmp_path: Path) -> None:
        db = await _fresh_db(tmp_path)
        try:
            now = datetime.now(timezone.utc)
            job = DownloadJob(
                id="j1", channel_id="C1", author="A", title="T",
                message_ids=[1, 2, 3], total=3,
                destination_dir="/tmp/x",
                created_at=now, updated_at=now,
            )
            await db.enqueue_download(job)
            nxt = await db.next_pending_download()
            assert nxt is not None
            assert nxt.id == "j1"
            assert nxt.message_ids == [1, 2, 3]
            await db.mark_download_status("j1", "in_progress", current_index=1, current_pct=50.0)
            await db.mark_download_status("j1", "done", overall_pct=100.0)
            async with db.conn.execute("SELECT status FROM download_jobs WHERE id='j1'") as cur:
                row = await cur.fetchone()
            assert row[0] == "done"
        finally:
            await db.close()

    async def test_reconcile_orphans(self, tmp_path: Path) -> None:
        db = await _fresh_db(tmp_path)
        try:
            now = datetime.now(timezone.utc)
            for jid, status in [("a", "in_progress"), ("b", "queued"), ("c", "done")]:
                j = DownloadJob(
                    id=jid, channel_id="C", author="A", title="T",
                    message_ids=[1], status=status, total=1,
                    destination_dir="/x",
                    created_at=now, updated_at=now,
                )
                await db.enqueue_download(j)
            n = await db.reconcile_orphans()
            assert n == 2
            async with db.conn.execute("SELECT id, status FROM download_jobs ORDER BY id") as cur:
                rows = await cur.fetchall()
            status_by = {r[0]: r[1] for r in rows}
            assert status_by == {"a": "failed", "b": "failed", "c": "done"}
        finally:
            await db.close()
