"""Migrate legacy JSON state files into the unified bjj.db.

Usage:
    python -m scripts.migrate_json_to_db [--config-dir /data/config] [--dry-run]

Imports (idempotent — existing DB rows are preserved):
    - settings.json           → settings table
    - background_jobs.json    → background_jobs table
    - jobs.json               → legacy_jobs table
    - library.json            → (reserved for Paso 3)
    - pipeline_history.json   → (reserved for Paso 3)

After successful import, each JSON file is renamed to <name>.json.bak.
Idempotente: ejecuciones posteriores son no-op (los archivos ya están
renombrados a .bak).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ossflow_service_kit.db import init_db, session_scope
from ossflow_service_kit.db.models import BackgroundJob as BackgroundJobRow
from ossflow_service_kit.db.models import Setting

# Asegura que LegacyJobRow se registre en Base.metadata antes de init_db().
from ossflow_api.modules.jobs.db import LegacyJobRow

log = logging.getLogger("migrate_json_to_db")


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Background jobs (cleanup_scan, duplicates_scan, etc.)
# ---------------------------------------------------------------------------


def migrate_background_jobs(config_dir: Path, *, dry_run: bool) -> int:
    """Importa background_jobs.json en la tabla background_jobs.

    Idempotente: filas con id ya presente en BD se preservan. El JSON se
    renombra a ``.bak`` solo si se importó al menos una entrada.
    """
    src = config_dir / "background_jobs.json"
    if not src.exists():
        log.info("background_jobs.json not found — skipping")
        return 0
    try:
        raw = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("background_jobs.json unreadable: %s", exc)
        return 0
    if not isinstance(raw, list):
        log.error("background_jobs.json is not an array — skipping")
        return 0

    imported = 0
    with session_scope() as s:
        existing = {r.id for r in s.query(BackgroundJobRow.id).all()}
        for d in raw:
            if not isinstance(d, dict):
                continue
            job_id = d.get("id")
            if not job_id or job_id in existing:
                continue
            if dry_run:
                log.info("[dry-run] would import background_jobs.%s", job_id)
            else:
                payload = {
                    "progress": d.get("progress"),
                    "message": d.get("message", ""),
                    "params": d.get("params", {}),
                }
                s.add(BackgroundJobRow(
                    id=job_id,
                    type=d.get("type", "unknown"),
                    status=d.get("status", "failed"),
                    payload=json.dumps(payload, ensure_ascii=False),
                    result=json.dumps(d["result"]) if d.get("result") else None,
                    error=d.get("error"),
                    created_at=_parse_dt(d.get("created_at")),
                    finished_at=_parse_dt(d.get("completed_at")),
                ))
            imported += 1

    if imported and not dry_run:
        backup = src.with_suffix(".json.bak")
        src.rename(backup)
        log.info("background_jobs.json → %s (imported %d jobs)", backup, imported)
    return imported


# ---------------------------------------------------------------------------
# Legacy jobs (chapter, subtitles, dubbing, elevenlabs)
# ---------------------------------------------------------------------------


def migrate_legacy_jobs(config_dir: Path, *, dry_run: bool) -> int:
    """Importa jobs.json en la tabla legacy_jobs.

    El JSON legacy es un dict ``{job_id: {job_id, job_type, video_path,
    status, ...}}`` (output de ``api.jobs_store.JobsStore``). Idempotente
    igual que las otras migraciones.
    """
    src = config_dir / "jobs.json"
    if not src.exists():
        log.info("jobs.json not found — skipping")
        return 0
    try:
        raw = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("jobs.json unreadable: %s", exc)
        return 0
    if not isinstance(raw, dict):
        log.error("jobs.json is not an object — skipping")
        return 0

    imported = 0
    with session_scope() as s:
        existing = {r.job_id for r in s.query(LegacyJobRow.job_id).all()}
        for jid, d in raw.items():
            if jid in existing:
                continue
            if not isinstance(d, dict):
                continue
            video_path = d.get("video_path") or ""
            if not video_path:
                # Sin video_path el job no es válido; saltar.
                log.warning("legacy job %s sin video_path — skipping", jid)
                continue
            if dry_run:
                log.info("[dry-run] would import legacy_jobs.%s", jid)
            else:
                s.add(LegacyJobRow(
                    job_id=jid,
                    job_type=d.get("job_type", "unknown"),
                    video_path=video_path,
                    status=d.get("status", "failed"),
                    progress=d.get("progress"),
                    message=d.get("message"),
                    result=json.dumps(d["result"]) if d.get("result") else None,
                    created_at=_parse_dt(d.get("created_at")),
                    completed_at=_parse_dt(d.get("completed_at")),
                ))
            imported += 1

    if imported and not dry_run:
        backup = src.with_suffix(".json.bak")
        src.rename(backup)
        log.info("jobs.json → %s (imported %d legacy jobs)", backup, imported)
    return imported


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="/data/config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    init_db()
    total = 0
    total += migrate_settings(args.config_dir, dry_run=args.dry_run)
    total += migrate_background_jobs(args.config_dir, dry_run=args.dry_run)
    total += migrate_legacy_jobs(args.config_dir, dry_run=args.dry_run)
    log.info("Migration complete — %d entries processed", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
