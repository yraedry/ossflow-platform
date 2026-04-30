"""Persistencia debounced del historial de pipelines.

Migrado de ``api/pipeline.py`` en T_LATE_2.2.

Las funciones reciben ``pipelines`` (dict mutable) y ``history_path``
(Path) explícitamente. El shim ``api/pipeline.py`` mantiene los
globals ``_pipelines`` y ``HISTORY_FILE`` por compat con tests que
parchean esos atributos via ``monkeypatch.setattr(pmod, ...)`` y
delega aquí pasándolos como argumentos.

El estado de debounce (``_save_lock``, ``_save_last_write``,
``_save_timer``) sí vive en este módulo: es interno al writer y los
tests no lo parchean.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .schemas import PipelineInfo, StepInfo, StepStatus, serialize_pipeline

log = logging.getLogger(__name__)

SAVE_MIN_INTERVAL = 2.0
_save_lock = threading.Lock()
_save_last_write = 0.0
_save_timer: Optional[threading.Timer] = None


def write_history_sync(
    pipelines: dict[str, PipelineInfo],
    history_path: Path,
) -> None:
    """Disk write síncrono (corre fuera del event loop)."""
    global _save_last_write
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        items = sorted(pipelines.values(), key=lambda p: p.created_at, reverse=True)[:200]
        payload = json.dumps(
            [serialize_pipeline(p) for p in items],
            indent=2,
            ensure_ascii=False,
        )
        history_path.write_text(payload, encoding="utf-8")
        _save_last_write = time.monotonic()
    except OSError as exc:
        log.warning("Failed to save pipeline history: %s", exc)


def save_history(
    pipelines: dict[str, PipelineInfo],
    history_path: Path,
) -> None:
    """Save no-bloqueante, debounced.

    Combina *fire-and-forget thread* + *debounce (2 s)*. Un thread daemon
    ejecuta serialización + write para no bloquear el event loop. Las
    ráfagas se coalescing schedulando una sola Timer trailing si el último
    write fue hace < 2 s.
    """
    global _save_timer
    now = time.monotonic()
    with _save_lock:
        elapsed = now - _save_last_write
        if elapsed >= SAVE_MIN_INTERVAL:
            if _save_timer is not None:
                _save_timer.cancel()
                _save_timer = None
            threading.Thread(
                target=write_history_sync,
                args=(pipelines, history_path),
                daemon=True,
            ).start()
        else:
            if _save_timer is None or not _save_timer.is_alive():
                delay = SAVE_MIN_INTERVAL - elapsed
                t = threading.Timer(
                    delay, write_history_sync, args=(pipelines, history_path),
                )
                t.daemon = True
                _save_timer = t
                t.start()


def load_history(
    pipelines: dict[str, PipelineInfo],
    history_path: Path,
) -> None:
    """Hidrata ``pipelines`` desde el JSON. Marca como FAILED los que
    estaban RUNNING/PENDING (proceso reiniciado a mitad de run)."""
    if not history_path.exists():
        return
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to load pipeline history: %s", exc)
        return
    for d in data:
        steps = [
            StepInfo(
                name=s["name"],
                status=StepStatus(s.get("status", "pending")),
                progress=s.get("progress", 0.0),
                message=s.get("message", ""),
                started_at=s.get("started_at"),
                completed_at=s.get("completed_at"),
                diff=s.get("diff"),
            )
            for s in d.get("steps", [])
        ]
        p = PipelineInfo(
            pipeline_id=d["pipeline_id"],
            path=d["path"],
            steps=steps,
            options=d.get("options", {}),
            status=StepStatus(d.get("status", "pending")),
            current_step=d.get("current_step", 0),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            completed_at=d.get("completed_at"),
        )
        if p.status in (StepStatus.RUNNING, StepStatus.PENDING):
            p.status = StepStatus.FAILED
            p.completed_at = p.completed_at or datetime.now(timezone.utc).isoformat()
            for s in p.steps:
                if s.status == StepStatus.RUNNING:
                    s.status = StepStatus.FAILED
        pipelines[p.pipeline_id] = p
