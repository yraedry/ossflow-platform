"""DTOs y constantes del módulo pipeline.

Migrado de ``api/pipeline.py`` en T_LATE_2.1. Contenido literal —
solo el split por archivo. Los tests parchean por nombre desde
``api.pipeline``, así que el shim re-exporta todos estos símbolos
sin alteración.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


VALID_STEPS = {"chapters", "subtitles", "translate", "dubbing"}
STEP_ORDER = ["chapters", "subtitles", "translate", "dubbing"]


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class StepInfo:
    name: str
    status: StepStatus = StepStatus.PENDING
    progress: float = 0.0
    message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    diff: Optional[dict] = None
    # Per-step options snapshot (e.g. dubbing_mode=True for translate runs
    # that generate .dub.es.srt). Lets the UI distinguish "Traducción" from
    # "Guion doblaje" even though both are the same backend step.
    options: dict = field(default_factory=dict)


@dataclass
class PipelineInfo:
    pipeline_id: str
    path: str
    steps: list[StepInfo]
    options: dict = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    current_step: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    # After `chapters` succeeds we redirect subsequent steps here — the Season
    # folder containing the freshly-split chapters. None until chapters runs.
    chained_path: Optional[str] = None
    log_buffer: list[dict] = field(default_factory=list)
    # Monotonic sequence counter for events (used for client-side dedupe).
    event_seq: int = 0


@dataclass
class BatchInfo:
    """A multi-season batch — wraps N pipelines run sequentially server-side.

    Lives entirely in-memory (intentionally not persisted): on container restart
    in-flight pipelines are already marked FAILED on history reload, so
    resuming a batch would mis-report state. The frontend treats a missing
    batch_id the same as 'finished' which is the correct behavior after a
    restart.
    """
    batch_id: str
    name: str
    paths: list[str]
    steps: list[str]
    options: dict
    continue_on_fail: bool = True
    status: StepStatus = StepStatus.PENDING
    current_index: int = 0
    pipeline_ids: list[str] = field(default_factory=list)
    last_error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None


def serialize_pipeline(p: PipelineInfo) -> dict[str, Any]:
    return {
        "pipeline_id": p.pipeline_id,
        "path": p.path,
        "options": p.options,
        "status": p.status.value,
        "current_step": p.current_step,
        "created_at": p.created_at,
        "completed_at": p.completed_at,
        "steps": [
            {
                "name": s.name,
                "status": s.status.value,
                "progress": s.progress,
                "message": s.message,
                "started_at": s.started_at,
                "completed_at": s.completed_at,
                "diff": s.diff,
            }
            for s in p.steps
        ],
    }


def serialize_batch(b: BatchInfo) -> dict[str, Any]:
    return {
        "batch_id": b.batch_id,
        "name": b.name,
        "paths": b.paths,
        "steps": b.steps,
        "options": b.options,
        "continue_on_fail": b.continue_on_fail,
        "status": b.status.value,
        "current_index": b.current_index,
        "total": len(b.paths),
        "pipeline_ids": b.pipeline_ids,
        "last_error": b.last_error,
        "created_at": b.created_at,
        "completed_at": b.completed_at,
    }
