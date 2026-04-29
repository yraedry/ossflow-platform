"""aiosqlite wrapper for telegram-fetcher cache.

Schema DDL matches architecture section 3. Migrations are idempotent; the
schema_version table holds the applied migration id so future bumps are safe.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiosqlite

from .models import DownloadJob, InstructionalGroup, MediaItem


# ---------------------------------------------------------------------------
# Canonical keys for grouping (author/title dedupe)
# ---------------------------------------------------------------------------


_AUTHOR_SUFFIX_RE = re.compile(
    r"\s+(?:volume|vol|part|chapter|ch|episode|ep)\.?(?:\s*\d+)?\s*$",
    re.IGNORECASE,
)
_BARE_VOLPART_RE = re.compile(
    r"^(?:volume|vol|part|chapter|ch|episode|ep)\.?(?:\s*\d+)?$",
    re.IGNORECASE,
)

_TITLE_TRAILING_NUM_RE = re.compile(r"\s*\d+\s*$")
_TITLE_VOLPART_PARENS_RE = re.compile(
    r"\s*[\(\[]\s*(?:volume|vol|part|chapter|ch|episode|ep)\.?\s*\d*\s*[\)\]]\s*",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _canonical_author_key(name: Optional[str]) -> str:
    """Normalized grouping key for an author name.

    Lowercases, strips accents/punctuation, and removes trailing
    ``Vol``/``Part``/``Volume``/``Chapter`` (optionally followed by a number)
    so that ``"Gordon Ryan"`` and ``"Gordon Ryan Vol"`` collapse to the same
    key.
    """
    if not name:
        return ""
    t = _strip_accents(name).lower().strip()
    if _BARE_VOLPART_RE.match(t):
        return ""
    # Repeatedly strip trailing volume-ish suffixes.
    prev = None
    while prev != t:
        prev = t
        t = _AUTHOR_SUFFIX_RE.sub("", t).strip()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    if _BARE_VOLPART_RE.match(t):
        return ""
    return t


def _canonical_title_key(title: Optional[str]) -> str:
    """Normalized grouping key for a title.

    Lowercases, strips accents/punctuation, removes ``(Vol N)`` /
    ``[Part N]`` residues and any trailing bare number so that chapter
    numbers leaking into the title don't create duplicate groups.
    """
    if not title:
        return ""
    t = _strip_accents(title).lower().strip()
    t = _TITLE_VOLPART_PARENS_RE.sub(" ", t)
    prev = None
    while prev != t:
        prev = t
        t = _TITLE_TRAILING_NUM_RE.sub("", t).strip()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


DEFAULT_DB_PATH = "/data/db/bjj.db"
CURRENT_SCHEMA_VERSION = 2


_SCHEMA_STATEMENTS: Sequence[str] = (
    """
    CREATE TABLE IF NOT EXISTS media (
      channel_id      TEXT    NOT NULL,
      message_id      INTEGER NOT NULL,
      caption         TEXT,
      filename        TEXT,
      size_bytes      INTEGER NOT NULL DEFAULT 0,
      mime_type       TEXT,
      date            TEXT    NOT NULL,
      author          TEXT,
      title           TEXT,
      chapter_num     INTEGER,
      manual_metadata INTEGER NOT NULL DEFAULT 0,
      available       INTEGER NOT NULL DEFAULT 1,
      downloaded_path TEXT,
      PRIMARY KEY (channel_id, message_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_media_channel_date ON media(channel_id, date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_media_author       ON media(author)",
    "CREATE INDEX IF NOT EXISTS idx_media_group        ON media(author, title, chapter_num)",
    """
    CREATE TABLE IF NOT EXISTS download_jobs (
      id              TEXT PRIMARY KEY,
      channel_id      TEXT NOT NULL,
      author          TEXT NOT NULL,
      title           TEXT NOT NULL,
      message_ids     TEXT NOT NULL,
      status          TEXT NOT NULL,
      current_index   INTEGER NOT NULL DEFAULT 0,
      total           INTEGER NOT NULL,
      current_pct     REAL NOT NULL DEFAULT 0,
      overall_pct     REAL NOT NULL DEFAULT 0,
      destination_dir TEXT NOT NULL,
      error           TEXT,
      created_at      TEXT NOT NULL,
      updated_at      TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON download_jobs(status, created_at)",
    """
    CREATE TABLE IF NOT EXISTS channels (
      channel_id      TEXT PRIMARY KEY,
      username        TEXT NOT NULL,
      title           TEXT,
      last_sync_at    TEXT,
      last_message_id INTEGER
    )
    """,
    "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)",
)


def _resolve_db_path(path: Optional[str]) -> str:
    # Unified DB: prefer BJJ_DB_PATH (shared with processor-api),
    # fall back to legacy TG_CACHE_DB, then default.
    return (
        path
        or os.environ.get("BJJ_DB_PATH")
        or os.environ.get("TG_CACHE_DB")
        or DEFAULT_DB_PATH
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _media_from_row(r: aiosqlite.Row) -> MediaItem:
    # thumbnail_path is optional (added in schema v2)
    try:
        thumb = r["thumbnail_path"]
    except (IndexError, KeyError):
        thumb = None
    return MediaItem(
        channel_id=r["channel_id"],
        message_id=r["message_id"],
        caption=r["caption"],
        filename=r["filename"],
        size_bytes=r["size_bytes"] or 0,
        mime_type=r["mime_type"],
        date=datetime.fromisoformat(r["date"]),
        author=r["author"],
        title=r["title"],
        chapter_num=r["chapter_num"],
        manual_metadata=bool(r["manual_metadata"]),
        available=bool(r["available"]),
        downloaded_path=r["downloaded_path"],
        thumbnail_path=thumb,
    )


class Database:
    """Async wrapper around aiosqlite. Keep a single instance per process."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = _resolve_db_path(path)
        self._conn: Optional[aiosqlite.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        for stmt in _SCHEMA_STATEMENTS:
            await self._conn.execute(stmt)
        # --- v2 migration: add media.thumbnail_path (idempotent) ---
        try:
            await self._conn.execute(
                "ALTER TABLE media ADD COLUMN thumbnail_path TEXT"
            )
        except aiosqlite.OperationalError:
            # Column already exists — SQLite raises "duplicate column name".
            pass
        # --- v3 migration: add channels.noforwards (content protection flag) ---
        try:
            await self._conn.execute(
                "ALTER TABLE channels ADD COLUMN noforwards INTEGER DEFAULT 0"
            )
        except aiosqlite.OperationalError:
            pass
        # Register schema version idempotently.
        async with self._conn.execute("SELECT MAX(version) AS v FROM schema_version") as cur:
            row = await cur.fetchone()
        current = row["v"] if row and row["v"] is not None else 0
        if current < CURRENT_SCHEMA_VERSION:
            await self._conn.execute(
                "INSERT OR IGNORE INTO schema_version(version) VALUES (?)",
                (CURRENT_SCHEMA_VERSION,),
            )
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized; call init() first")
        return self._conn

    # ------------------------------------------------------------------
    # Media
    # ------------------------------------------------------------------

    async def upsert_media(self, item: MediaItem) -> None:
        sql = """
        INSERT INTO media(channel_id, message_id, caption, filename, size_bytes, mime_type,
                          date, author, title, chapter_num, manual_metadata, available,
                          downloaded_path)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(channel_id, message_id) DO UPDATE SET
            caption         = excluded.caption,
            filename        = excluded.filename,
            size_bytes      = excluded.size_bytes,
            mime_type       = excluded.mime_type,
            date            = excluded.date,
            author          = CASE WHEN media.manual_metadata = 1 THEN media.author ELSE excluded.author END,
            title           = CASE WHEN media.manual_metadata = 1 THEN media.title  ELSE excluded.title  END,
            chapter_num     = CASE WHEN media.manual_metadata = 1 THEN media.chapter_num ELSE excluded.chapter_num END,
            available       = excluded.available,
            downloaded_path = COALESCE(excluded.downloaded_path, media.downloaded_path)
        """
        await self.conn.execute(
            sql,
            (
                item.channel_id,
                item.message_id,
                item.caption,
                item.filename,
                item.size_bytes,
                item.mime_type,
                item.date.isoformat(),
                item.author,
                item.title,
                item.chapter_num,
                1 if item.manual_metadata else 0,
                1 if item.available else 0,
                item.downloaded_path,
            ),
        )
        await self.conn.commit()

    async def list_media(
        self,
        channel_id: Optional[str] = None,
        *,
        limit: int = 500,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> List[MediaItem]:
        clauses: List[str] = []
        params: List[Any] = []
        if channel_id is not None:
            clauses.append("channel_id = ?")
            params.append(channel_id)
        if search:
            clauses.append("(caption LIKE ? OR filename LIKE ? OR title LIKE ? OR author LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM media {where} ORDER BY date DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        async with self.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_media_from_row(r) for r in rows]

    async def group_by_author(self, channel_id: Optional[str] = None) -> List[InstructionalGroup]:
        clauses = ["author IS NOT NULL", "title IS NOT NULL"]
        params: List[Any] = []
        if channel_id is not None:
            clauses.append("channel_id = ?")
            params.append(channel_id)
        where = "WHERE " + " AND ".join(clauses)
        # Pull raw rows; we do the grouping in Python so we can apply the
        # canonical-key normalization (strip trailing "Vol"/"Part" residues,
        # unify accents/case/punctuation). SQLite's GROUP BY is too literal
        # for this job.
        sql = f"""
            SELECT author, title, message_id, size_bytes, available,
                   downloaded_path, chapter_num, channel_id, thumbnail_path
            FROM media
            {where}
            ORDER BY author ASC, title ASC, message_id ASC
        """
        async with self.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()

        # key = (author_key, title_key)
        buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in rows:
            raw_author = r["author"]
            raw_title = r["title"]
            a_key = _canonical_author_key(raw_author)
            t_key = _canonical_title_key(raw_title)
            if not a_key or not t_key:
                # Skip rows whose normalized author/title collapses to empty
                # (e.g. author was just "Vol 1"). They should not form a group.
                continue
            key = (a_key, t_key)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = {
                    "author": raw_author,
                    "title": raw_title,
                    "author_len": len(raw_author or ""),
                    "title_len": len(raw_title or ""),
                    "ids": [],
                    "size": 0,
                    "available": True,
                    "dl": 0,
                    "first_chapter_num": None,
                    "first_message_id": None,
                    "first_channel_id": None,
                    "first_thumb": None,
                }
                buckets[key] = bucket
            # Track the "first chapter" preview (chapter_num=1 wins; otherwise
            # the row with the lowest message_id seen so far).
            try:
                row_chapter = r["chapter_num"]
            except (IndexError, KeyError):
                row_chapter = None
            try:
                row_thumb = r["thumbnail_path"]
            except (IndexError, KeyError):
                row_thumb = None
            try:
                row_chan = r["channel_id"]
            except (IndexError, KeyError):
                row_chan = None
            row_msg = int(r["message_id"])
            cur_chap = bucket["first_chapter_num"]
            cur_msg = bucket["first_message_id"]
            replace = False
            if cur_msg is None:
                replace = True
            elif row_chapter == 1 and cur_chap != 1:
                replace = True
            elif (cur_chap != 1) and row_msg < (cur_msg or row_msg + 1):
                replace = True
            if replace:
                bucket["first_chapter_num"] = row_chapter
                bucket["first_message_id"] = row_msg
                bucket["first_channel_id"] = row_chan
                bucket["first_thumb"] = row_thumb
            # Pick the shortest (= most normalized) display representation.
            if raw_author and len(raw_author) < bucket["author_len"]:
                bucket["author"] = raw_author
                bucket["author_len"] = len(raw_author)
            if raw_title and len(raw_title) < bucket["title_len"]:
                bucket["title"] = raw_title
                bucket["title_len"] = len(raw_title)
            bucket["ids"].append(int(r["message_id"]))
            bucket["size"] += int(r["size_bytes"] or 0)
            bucket["available"] = bucket["available"] and bool(r["available"])
            if r["downloaded_path"]:
                bucket["dl"] += 1

        groups: List[InstructionalGroup] = []
        for bucket in buckets.values():
            groups.append(
                InstructionalGroup(
                    author=bucket["author"],
                    title=bucket["title"],
                    chapter_count=len(bucket["ids"]),
                    total_size_bytes=bucket["size"],
                    available=bucket["available"],
                    message_ids=sorted(bucket["ids"]),
                    downloaded_chapters=bucket["dl"],
                    first_channel_id=bucket["first_channel_id"],
                    first_message_id=bucket["first_message_id"],
                    first_thumbnail_path=bucket["first_thumb"],
                )
            )
        groups.sort(key=lambda g: (g.author.lower(), g.title.lower()))
        return groups

    async def set_thumbnail_path(
        self,
        channel_id: str,
        message_id: int,
        thumbnail_path: Optional[str],
    ) -> None:
        """Persist thumbnail file path for a media row. No-op if row missing."""
        await self.conn.execute(
            "UPDATE media SET thumbnail_path = ? WHERE channel_id = ? AND message_id = ?",
            (thumbnail_path, channel_id, int(message_id)),
        )
        await self.conn.commit()

    async def get_thumbnail_path(
        self, channel_id: str, message_id: int
    ) -> Optional[str]:
        async with self.conn.execute(
            "SELECT thumbnail_path FROM media WHERE channel_id = ? AND message_id = ?",
            (channel_id, int(message_id)),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        try:
            return row["thumbnail_path"]
        except (IndexError, KeyError):
            return None

    async def update_metadata(
        self,
        channel_id: str,
        message_id: int,
        *,
        author: Optional[str],
        title: Optional[str],
        chapter_num: Optional[int],
        manual: bool = True,
    ) -> None:
        await self.conn.execute(
            """
            UPDATE media
               SET author = ?, title = ?, chapter_num = ?, manual_metadata = ?
             WHERE channel_id = ? AND message_id = ?
            """,
            (author, title, chapter_num, 1 if manual else 0, channel_id, int(message_id)),
        )
        await self.conn.commit()

    # ------------------------------------------------------------------
    # Channels
    # ------------------------------------------------------------------

    async def list_channels(self) -> List[Dict[str, Any]]:
        async with self.conn.execute(
            """
            SELECT c.channel_id, c.username, c.title, c.last_sync_at, c.last_message_id,
                   COALESCE(c.noforwards, 0) AS noforwards,
                   (SELECT COUNT(*) FROM media m WHERE m.channel_id = c.channel_id) AS media_count
              FROM channels c
             ORDER BY c.title COLLATE NOCASE ASC
            """
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "channel_id": r["channel_id"],
                "username": r["username"],
                "title": r["title"],
                "last_sync_at": r["last_sync_at"],
                "last_message_id": r["last_message_id"],
                "media_count": r["media_count"] or 0,
                "noforwards": bool(r["noforwards"]),
            }
            for r in rows
        ]

    async def get_channel_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        async with self.conn.execute(
            "SELECT channel_id, username, title FROM channels WHERE username = ? COLLATE NOCASE",
            (username,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return {"channel_id": row["channel_id"], "username": row["username"], "title": row["title"]}

    async def update_channel_title(self, channel_id: str, title: str) -> bool:
        cur = await self.conn.execute(
            "UPDATE channels SET title = ? WHERE channel_id = ?",
            (title, channel_id),
        )
        await self.conn.commit()
        return (cur.rowcount or 0) > 0

    async def list_media_thumbnails(self, channel_id: str) -> List[str]:
        async with self.conn.execute(
            "SELECT thumbnail_path FROM media WHERE channel_id = ? AND thumbnail_path IS NOT NULL",
            (channel_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [r["thumbnail_path"] for r in rows if r["thumbnail_path"]]

    async def delete_channel_cascade(self, channel_id: str) -> bool:
        """Remove channel row + all media + download jobs for it. Returns True if channel existed."""
        async with self.conn.execute(
            "SELECT 1 FROM channels WHERE channel_id = ?", (channel_id,)
        ) as cur:
            exists = await cur.fetchone() is not None
        if not exists:
            return False
        await self.conn.execute("DELETE FROM media WHERE channel_id = ?", (channel_id,))
        await self.conn.execute("DELETE FROM download_jobs WHERE channel_id = ?", (channel_id,))
        await self.conn.execute("DELETE FROM channels WHERE channel_id = ?", (channel_id,))
        await self.conn.commit()
        return True

    async def upsert_channel(
        self,
        channel_id: str,
        username: str,
        *,
        title: Optional[str] = None,
        last_message_id: Optional[int] = None,
        noforwards: Optional[bool] = None,
    ) -> None:
        await self.conn.execute(
            """
            INSERT INTO channels(channel_id, username, title, last_sync_at, last_message_id, noforwards)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(channel_id) DO UPDATE SET
                username = excluded.username,
                title    = COALESCE(excluded.title, channels.title),
                last_sync_at = excluded.last_sync_at,
                last_message_id = COALESCE(excluded.last_message_id, channels.last_message_id),
                noforwards = COALESCE(excluded.noforwards, channels.noforwards)
            """,
            (
                channel_id,
                username,
                title,
                _now(),
                last_message_id,
                (1 if noforwards else 0) if noforwards is not None else None,
            ),
        )
        await self.conn.commit()

    # ------------------------------------------------------------------
    # Download jobs
    # ------------------------------------------------------------------

    async def enqueue_download(self, job: DownloadJob) -> None:
        await self.conn.execute(
            """
            INSERT INTO download_jobs(id, channel_id, author, title, message_ids, status,
                                      current_index, total, current_pct, overall_pct,
                                      destination_dir, error, created_at, updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                job.id,
                job.channel_id,
                job.author,
                job.title,
                json.dumps(job.message_ids),
                job.status,
                job.current_index,
                job.total,
                job.current_pct,
                job.overall_pct,
                job.destination_dir,
                job.error,
                job.created_at.isoformat(),
                job.updated_at.isoformat(),
            ),
        )
        await self.conn.commit()

    async def next_pending_download(self) -> Optional[DownloadJob]:
        async with self.conn.execute(
            "SELECT * FROM download_jobs WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return DownloadJob(
            id=row["id"],
            channel_id=row["channel_id"],
            author=row["author"],
            title=row["title"],
            message_ids=json.loads(row["message_ids"] or "[]"),
            status=row["status"],
            current_index=row["current_index"],
            total=row["total"],
            current_pct=row["current_pct"],
            overall_pct=row["overall_pct"],
            destination_dir=row["destination_dir"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def mark_download_status(
        self,
        job_id: str,
        status: str,
        *,
        current_index: Optional[int] = None,
        current_pct: Optional[float] = None,
        overall_pct: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        sets = ["status = ?", "updated_at = ?"]
        params: List[Any] = [status, _now()]
        if current_index is not None:
            sets.append("current_index = ?")
            params.append(current_index)
        if current_pct is not None:
            sets.append("current_pct = ?")
            params.append(current_pct)
        if overall_pct is not None:
            sets.append("overall_pct = ?")
            params.append(overall_pct)
        if error is not None:
            sets.append("error = ?")
            params.append(error)
        params.append(job_id)
        sql = f"UPDATE download_jobs SET {', '.join(sets)} WHERE id = ?"
        await self.conn.execute(sql, params)
        await self.conn.commit()

    async def reconcile_orphans(self) -> int:
        """Mark any in_progress/queued job as failed at startup.

        Returns the number of reconciled rows.
        """
        cur = await self.conn.execute(
            """
            UPDATE download_jobs
               SET status = 'failed',
                   error  = COALESCE(error, 'orphaned by restart'),
                   updated_at = ?
             WHERE status IN ('in_progress', 'queued')
            """,
            (_now(),),
        )
        rowcount = cur.rowcount or 0
        await self.conn.commit()
        return int(rowcount)
