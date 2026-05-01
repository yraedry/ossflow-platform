"""Compat shim del módulo pipeline (T_LATE_2.7).

La lógica vive en ``ossflow_api.modules.pipeline.*``. Este shim mantiene
la API que los tests parchean por nombre desde ``api.pipeline``:

* State container (``_pipelines``, ``_batches``, ``_pipeline_subscribers``,
  ``_pipeline_tasks``, ``_pipeline_cancel``, ``_batch_tasks``,
  ``_batch_cancel``).
* Funciones públicas (``_run_pipeline``, ``_run_step``, ``_run_batch``,
  ``_emit``, ``_subscribe``, ``_unsubscribe``, ``_save_history``,
  ``_load_history``, ``_target_dir``, ``_compute_diff``,
  ``_detect_season_folder``, ``_chapter_has_*``, ``_season_already_*``,
  ``_client_and_payload``, ``_load_oracle_for_path``, ``_total_video_duration``,
  ``_duration_seconds``, ``_median``, ``_launch_pipeline_internal``,
  ``_refresh_scan_cache_for``, ``_flush_gpu_after_step``).
* Backend client factories (``splitter_client``, ``subs_client``,
  ``dubbing_client``, ``get_library_path``).
* Constantes (``HISTORY_FILE``, ``VALID_STEPS``, ``STEP_ORDER``,
  ``SIDECAR_NAME``).
* Re-export del ``router`` con todos los endpoints.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

# Backend clients + paths — los tests parchean estos nombres aquí
# (``monkeypatch.setattr(pipeline_module, "splitter_client", ...)``).
from api.backend_client import (  # noqa: F401
    BackendClient,
    BackendError,
    dubbing_client,
    splitter_client,
    subs_client,
)
from api.event_normalizer import normalize  # noqa: F401
from api.paths import to_container_path  # noqa: F401
from api.settings import CONFIG_DIR as _CONFIG_DIR  # noqa: F401
from api.settings import get_library_path  # noqa: F401

log = logging.getLogger(__name__)

HISTORY_FILE = _CONFIG_DIR / "pipeline_history.json"
SIDECAR_NAME = ".bjj-meta.json"


# ---------------------------------------------------------------------------
# Schemas (T_LATE_2.1)
# ---------------------------------------------------------------------------

from ossflow_api.modules.pipeline.schemas import (  # noqa: F401,E402
    BatchInfo,
    PipelineInfo,
    STEP_ORDER,
    StepInfo,
    StepStatus,
    VALID_STEPS,
    serialize_batch as _serialize_batch_new,
    serialize_pipeline as _serialize_new,
)


# ---------------------------------------------------------------------------
# State container — globals del shim. Los tests parchean / mutan los dicts.
# ---------------------------------------------------------------------------

_pipelines: dict[str, PipelineInfo] = {}
_batches: dict[str, BatchInfo] = {}
_batch_tasks: dict[str, asyncio.Task] = {}
_batch_cancel: dict[str, bool] = {}
# Per-pipeline list of subscriber queues (fan-out).
_pipeline_subscribers: dict[str, list[asyncio.Queue]] = {}
_pipeline_tasks: dict[str, asyncio.Task] = {}
_pipeline_cancel: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# History persistence (T_LATE_2.2)
# ---------------------------------------------------------------------------

from ossflow_api.modules.pipeline import history as _history_mod  # noqa: E402

_SAVE_MIN_INTERVAL = _history_mod.SAVE_MIN_INTERVAL


def _write_history_sync() -> None:
    _history_mod.write_history_sync(_pipelines, HISTORY_FILE)


def _save_history() -> None:
    _history_mod.save_history(_pipelines, HISTORY_FILE)


def _load_history() -> None:
    _history_mod.load_history(_pipelines, HISTORY_FILE)


_load_history()


# ---------------------------------------------------------------------------
# SSE primitives (T_LATE_2.5b)
# ---------------------------------------------------------------------------

from ossflow_api.modules.pipeline import store as _store_mod  # noqa: E402


def _subscribe(pipeline_id: str) -> asyncio.Queue:
    return _store_mod.subscribe(_pipeline_subscribers, pipeline_id)


def _unsubscribe(pipeline_id: str, q: asyncio.Queue) -> None:
    _store_mod.unsubscribe(_pipeline_subscribers, pipeline_id, q)


async def _emit(pipeline: PipelineInfo, queue: asyncio.Queue, event: dict) -> None:
    """Wrapper retrocompat — ``queue`` se ignora (legacy)."""
    await _store_mod.emit(pipeline, _pipeline_subscribers, event)


def _serialize(p: PipelineInfo) -> dict:
    return _serialize_new(p)


def _serialize_batch(b: BatchInfo) -> dict:
    return _serialize_batch_new(b)


# ---------------------------------------------------------------------------
# Skip-detection + diff (T_LATE_2.3)
# ---------------------------------------------------------------------------

from ossflow_api.modules.pipeline.skip_detector import (  # noqa: E402,F401
    CHAPTER_SE_RE as _CHAPTER_SE_RE,
    DUB_SUFFIXES as _DUB_SUFFIXES,
    chapter_has_en_subs as _chapter_has_en_subs,
    chapter_has_es_subs as _chapter_has_es_subs,
    chapter_is_dubbed as _chapter_is_dubbed,
    list_chapters as _list_chapters,
    season_already_dubbed as _season_already_dubbed,
    season_already_subbed_en as _season_already_subbed_en,
    season_already_subbed_es as _season_already_subbed_es,
)
from ossflow_api.modules.pipeline.diff import (  # noqa: E402,F401
    SEASON_DIR_RE as _SEASON_DIR_RE,
    VIDEO_EXTS as _VIDEO_EXTS,
    compute_diff as _compute_diff,
    detect_season_folder as _detect_season_folder,
    snapshot_dir as _snapshot_dir,
    target_dir as _target_dir,
)


# ---------------------------------------------------------------------------
# Backend dispatch (T_LATE_2.4)
# ---------------------------------------------------------------------------

from ossflow_api.modules.pipeline.backend_dispatch import (  # noqa: E402,F401
    client_and_payload as _client_and_payload_new,
    load_oracle_for_path as _load_oracle_for_path,
    load_voice_profile_for_path as _load_voice_profile_for_path,
)


def _client_and_payload(
    step_name: str,
    path: str,
    options: dict,
    chained_path: Optional[str] = None,
) -> tuple[BackendClient, dict, bool]:
    """Wrapper retrocompat — la lógica vive en
    ``modules/pipeline/backend_dispatch.py``. Tests parchean este
    símbolo en el shim."""
    return _client_and_payload_new(step_name, path, options, chained_path)


# ---------------------------------------------------------------------------
# ETA helpers (T_LATE_2.5a)
# ---------------------------------------------------------------------------

from ossflow_api.modules.pipeline.eta import (  # noqa: E402,F401
    duration_seconds as _duration_seconds,
    median as _median,
    total_video_duration as _total_video_duration,
)


# ---------------------------------------------------------------------------
# Runner (T_LATE_2.5c-d)
# ---------------------------------------------------------------------------

from ossflow_api.modules.pipeline import runner as _runner_mod  # noqa: E402


async def _run_step(
    pipeline: PipelineInfo,
    step_index: int,
    queue: asyncio.Queue,
) -> bool:
    return await _runner_mod.run_step(pipeline, step_index, queue)


async def _flush_gpu_after_step(
    pipeline: PipelineInfo, queue: asyncio.Queue,
) -> None:
    await _runner_mod.flush_gpu_after_step(pipeline, queue)


async def _run_pipeline(pipeline: PipelineInfo, queue: asyncio.Queue) -> None:
    await _runner_mod.run_pipeline(pipeline, queue)


async def _run_batch(batch: BatchInfo) -> None:
    await _runner_mod.run_batch(batch)


def _refresh_scan_cache_for(pipeline_path: str) -> None:
    _runner_mod.refresh_scan_cache_for(pipeline_path)


# ---------------------------------------------------------------------------
# Launch helper + Router (T_LATE_2.6)
# ---------------------------------------------------------------------------

from ossflow_api.modules.pipeline.router import (  # noqa: E402,F401
    launch_pipeline_internal as _launch_pipeline_internal,
    router,
)
