"""Tests de BackgroundJobsService.

Anti-flakiness: ``threading.Event`` cross-thread, ``_wait_until``,
timeouts cortos. Cada test usa BD efímera (tmp_path).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable

import pytest

from ossflow_service_kit.db import engine as eng_mod
from ossflow_service_kit.db import session as sess_mod

from ossflow_api.modules.jobs._internal.scheduler import JobsScheduler
from ossflow_api.modules.jobs.repositories.background import BackgroundJobsRepository
from ossflow_api.modules.jobs.services.background import BackgroundJobsService


def _wait_until(pred: Callable[[], bool], *, timeout: float = 2.0, step: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return pred()


@pytest.fixture
def svc(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    eng_mod.reset_engine()
    sess_mod.reset_factory()
    repo = BackgroundJobsRepository()
    scheduler = JobsScheduler()
    service = BackgroundJobsService(repo, scheduler)
    service.init()
    yield service
    eng_mod.reset_engine()
    sess_mod.reset_factory()


def test_init_is_idempotent(svc):
    """Llamar init dos veces no rompe ni duplica trabajo."""
    svc.init()
    svc.init()  # no-op
    assert svc._initialized is True


def test_submit_creates_job_and_runs_to_completion(svc):
    """Happy path: submit → ejecuta → COMPLETED con result."""
    finished = threading.Event()

    async def coro_factory(update_progress):
        update_progress(50.0, "halfway")
        return {"output": "ok"}

    job = svc.submit("cleanup_scan", coro_factory, {"path": "/x"})
    assert job.status == "queued"
    assert job.params == {"path": "/x"}

    # Espera a que termine.
    def is_done():
        nonlocal finished
        j = svc.get(job.id)
        if j and j.status in ("completed", "failed"):
            finished.set()
            return True
        return False

    assert _wait_until(is_done, timeout=2.0)

    final = svc.get(job.id)
    assert final.status == "completed"
    assert final.result == {"output": "ok"}
    assert final.progress == 100.0
    assert final.completed_at is not None


def test_submit_handles_exception_as_failed(svc):
    """Excepción en el coroutine → status FAILED con error poblado."""
    async def coro_factory(update_progress):
        raise ValueError("boom")

    job = svc.submit("cleanup_scan", coro_factory, {})

    assert _wait_until(
        lambda: (j := svc.get(job.id)) and j.status == "failed",
        timeout=2.0,
    )
    final = svc.get(job.id)
    assert final.status == "failed"
    assert "ValueError" in final.error
    assert "boom" in final.error
    assert final.completed_at is not None


def test_submit_returns_non_dict_wraps_in_value_key(svc):
    """Si el coroutine devuelve algo no-dict, el service lo envuelve como
    ``{"value": ...}`` (compat con el contrato legacy)."""
    async def coro_factory(update_progress):
        return "string-result"

    job = svc.submit("cleanup_scan", coro_factory, {})
    assert _wait_until(
        lambda: (j := svc.get(job.id)) and j.status == "completed",
        timeout=2.0,
    )
    final = svc.get(job.id)
    assert final.result == {"value": "string-result"}


def test_submit_persists_status_transitions(svc):
    """El job se persiste en BD: existe antes de terminar y al terminar."""
    started = threading.Event()
    block = threading.Event()

    async def coro_factory(update_progress):
        started.set()
        # Espera bloqueante (en su propio loop) hasta señal externa.
        while not block.is_set():
            await asyncio.sleep(0.01)
        return {"ok": True}

    job = svc.submit("cleanup_scan", coro_factory, {})

    # En cuanto el thread arranca, el job está en BD con status running.
    assert started.wait(timeout=1.0)
    assert _wait_until(
        lambda: (j := svc.get(job.id)) and j.status == "running",
        timeout=1.0,
    )

    # Liberamos el coroutine.
    block.set()

    assert _wait_until(
        lambda: (j := svc.get(job.id)) and j.status == "completed",
        timeout=2.0,
    )


def test_list_all_after_multiple_submits(svc):
    async def quick(update_progress):
        return {}

    j1 = svc.submit("type_a", quick, {})
    j2 = svc.submit("type_b", quick, {})
    j3 = svc.submit("type_a", quick, {})

    # Espera a que todos terminen.
    assert _wait_until(
        lambda: all(
            (j := svc.get(jid)) and j.status == "completed"
            for jid in (j1.id, j2.id, j3.id)
        ),
        timeout=2.0,
    )

    all_jobs = svc.list_all()
    assert len(all_jobs) == 3

    type_a = svc.list_all(type_filter="type_a")
    assert {j.id for j in type_a} == {j1.id, j3.id}
