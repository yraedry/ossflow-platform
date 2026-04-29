"""Migrate legacy telegram.db into the unified bjj.db.

Usage:
    python -m scripts.migrate_to_unified_db \
        --source /data/cache/telegram.db \
        --target /data/db/bjj.db [--dry-run]

Strategy: telegram-fetcher tables (media, download_jobs, channels,
schema_version) are disjoint from processor-api tables in the unified DB,
so we ATTACH the legacy file and INSERT OR IGNORE row-by-row. Idempotent.

After success, source is renamed to <source>.bak.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

log = logging.getLogger("telegram_migrate")

TABLES = ("channels", "media", "download_jobs", "schema_version")


def migrate(source: Path, target: Path, *, dry_run: bool) -> int:
    if not source.exists():
        log.info("Source %s not found — nothing to migrate", source)
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)

    # Ensure target exists (tables will be created by telegram-fetcher init
    # on first run, but we can also just copy the file if target is absent).
    if not target.exists():
        if dry_run:
            log.info("[dry-run] would copy %s → %s (target absent)", source, target)
            return 0
        shutil.copy2(source, target)
        log.info("Copied legacy DB → %s", target)
        source.rename(source.with_suffix(source.suffix + ".bak"))
        return 1

    # Target exists — merge row by row.
    conn = sqlite3.connect(target)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"ATTACH DATABASE ? AS legacy", (str(source),))

    merged = 0
    try:
        for table in TABLES:
            try:
                cur = conn.execute(f"SELECT COUNT(*) FROM legacy.{table}")
                count = cur.fetchone()[0]
            except sqlite3.OperationalError:
                log.info("legacy table %s missing — skip", table)
                continue
            if dry_run:
                log.info("[dry-run] would merge %d rows from legacy.%s", count, table)
                merged += count
                continue
            # Idempotent: INSERT OR IGNORE preserves existing rows in target.
            conn.execute(
                f"INSERT OR IGNORE INTO {table} SELECT * FROM legacy.{table}"
            )
            merged += count
            log.info("Merged %d rows from legacy.%s", count, table)
        if not dry_run:
            conn.commit()
    finally:
        conn.execute("DETACH DATABASE legacy")
        conn.close()

    if merged and not dry_run:
        backup = source.with_suffix(source.suffix + ".bak")
        source.rename(backup)
        log.info("Legacy DB renamed to %s", backup)
    return merged


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="/data/cache/telegram.db", type=Path)
    p.add_argument("--target", default="/data/db/bjj.db", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    merged = migrate(args.source, args.target, dry_run=args.dry_run)
    log.info("Migration done — %d rows processed", merged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
