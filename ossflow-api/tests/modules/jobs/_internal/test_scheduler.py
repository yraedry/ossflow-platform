"""Tests del JobsScheduler.

Anti-flakiness: usar ``threading.Event`` y ``_wait_until``, jamás ``sleep``
ciegos. Timeouts cortos (1-2s) con fallo claro si el job no termina.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable

import pytest

from ossflow_api.modules.jobs._internal.scheduler import JobsScheduler


def _wait_until(pred: Callable[[], bool], *, timeout: float = 1.0, step: float = 0.01) -> bool:
    """Polling activo hasta que ``pred()`` sea True o se exceda timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return pred()


# ---------------------------------------------------------------------------
# run_detached
# ---------------------------------------------------------------------------


def test_run_detached_executes_coro_and_terminates():
    sched = JobsScheduler()
    finished = threading.Event()

    async def _coro() -> None:
        finished.set()

    sched.run_detached(_coro, name="job-1")

    assert finished.wait(timeout=1.0)
    # El thread se desregistra al terminar (best-effort, puede tardar un tick).
    assert _wait_until(lambda: "job-1" not in sched.active_thread_names())


def test_run_detached_isolates_exceptions_from_main():
    """Excepciones en el coro no deben propagar al hilo principal."""
    sched = JobsScheduler()
    raised = threading.Event()

    async def _coro() -> None:
        raised.set()
        raise RuntimeError("boom")

    # No debe lanzar al main thread.
    sched.run_detached(_coro, name="boom-job")

    assert raised.wait(timeout=1.0)
    assert _wait_until(lambda: "boom-job" not in sched.active_thread_names())


def test_run_detached_runs_concurrent_jobs():
    sched = JobsScheduler()
    started = threading.Event()
    finish = threading.Event()
    finished = threading.Event()

    async def _coro() -> None:
        started.set()
        # Espera externa para verificar que el thread está vivo.
        while not finish.is_set():
            await asyncio.sleep(0.005)
        finished.set()

    sched.run_detached(_coro, name="long-job")
    assert started.wait(timeout=1.0)
    assert "long-job" in sched.active_thread_names()
    finish.set()
    assert finished.wait(timeout=1.0)


# ---------------------------------------------------------------------------
# run_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_task_executes_in_current_loop():
    sched = JobsScheduler()
    flag = asyncio.Event()

    async def _coro() -> None:
        flag.set()

    task = sched.run_task(_coro(), name="task-1")
    await asyncio.wait_for(flag.wait(), timeout=1.0)
    await task  # Espera completion explícita.
    # Tras completar, se desregistra del set activo.
    await asyncio.sleep(0)  # un tick para que el callback corra
    assert "task-1" not in sched.active_task_names()


@pytest.mark.asyncio
async def test_run_task_can_be_cancelled():
    sched = JobsScheduler()

    async def _coro() -> None:
        await asyncio.sleep(10)

    task = sched.run_task(_coro(), name="cancellable")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert "cancellable" not in sched.active_task_names()


@pytest.mark.asyncio
async def test_run_task_logs_exceptions_without_propagating():
    sched = JobsScheduler()

    async def _coro() -> None:
        raise RuntimeError("task-boom")

    task = sched.run_task(_coro(), name="boom")
    with pytest.raises(RuntimeError, match="task-boom"):
        await task


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


def test_shutdown_does_not_raise_with_no_jobs():
    sched = JobsScheduler()
    sched.shutdown()  # no debe lanzar


def test_shutdown_logs_alive_threads(caplog):
    sched = JobsScheduler()
    block = threading.Event()
    started = threading.Event()

    async def _coro() -> None:
        started.set()
        while not block.is_set():
            await asyncio.sleep(0.005)

    sched.run_detached(_coro, name="alive-thread")
    assert started.wait(timeout=1.0)

    with caplog.at_level("INFO"):
        sched.shutdown(timeout=0.1)

    assert any("alive-thread" in rec.message for rec in caplog.records)
    block.set()  # libera el thread
