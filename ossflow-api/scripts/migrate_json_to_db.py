"""Migrate legacy JSON state files into the unified bjj.db.

Usage:
    python -m scripts.migrate_json_to_db [--config-dir /data/config] [--dry-run]

Imports (idempotent — existing DB rows are preserved):
    - settings.json           → settings table
    - jobs.json               → (reserved for Paso 3)
    - library.json            → (reserved for Paso 3)
    - background_jobs.json    → (reserved for Paso 3)
    - pipeline_history.json   → (reserved for Paso 3)

After successful import, each JSON file is renamed to <name>.json.bak.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bjj_service_kit.db import init_db, session_scope
from bjj_service_kit.db.models import Setting

log = logging.getLogger("migrate_json_to_db")


def migrate_settings(config_dir: Path, *, dry_run: bool) -> int:
    src = config_dir / "settings.json"
    if not src.exists():
        log.info("settings.json not found — skipping")
        return 0
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("settings.json unreadable: %s", exc)
        return 0
    if not isinstance(data, dict):
        log.error("settings.json is not an object — skipping")
        return 0

    imported = 0
    with session_scope() as s:
        existing = {row.key for row in s.query(Setting).all()}
        for k, v in data.items():
            if k in existing:
                continue
            if dry_run:
                log.info("[dry-run] would import settings.%s", k)
            else:
                s.add(Setting(key=k, value=json.dumps(v, ensure_ascii=False)))
            imported += 1

    if imported and not dry_run:
        backup = src.with_suffix(".json.bak")
        src.rename(backup)
        log.info("settings.json → %s (imported %d keys)", backup, imported)
    return imported


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="/data/config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    init_db()
    total = 0
    total += migrate_settings(args.config_dir, dry_run=args.dry_run)
    log.info("Migration complete — %d entries processed", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
