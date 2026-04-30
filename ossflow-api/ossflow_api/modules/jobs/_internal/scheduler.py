"""Scheduler de jobs: lanza coroutines en background con dos modos.

* ``run_detached``: ``threading.Thread`` con su propio ``asyncio.run``.
  Caso de ``BackgroundJobsService`` — el job sobrevive al ciclo de vida
  del request HTTP (clave en ``cleanup_scan`` que dura minutos, y para
  que ``TestClient`` no destruya el loop antes de tiempo).

* ``run_task``: ``asyncio.create_task`` en el loop actual. Caso de
  ``LegacyJobsService`` — el job vive en el loop del request, lo que
  permite SSE compartido sin sincronización cross-thread.

Sin estado de jobs: este componente solo gestiona la *forma de ejecutar*.
La tracking de qué jobs están vivos lo hacen los servicios.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

CoroFactory = Callable[[], Awaitable[Any]]


class JobsScheduler:
    """Lanza coroutines en background con seguimiento mínimo de threads/tasks."""

    def __init__(self) -> None:
        self._threads: dict[str, threading.Thread] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = threading.Lock()

    # --- modo detached (thread propio + loop propio) -----------------------

    def run_detached(self, coro_factory: CoroFactory, *, name: str) -> None:
        """Crea un ``threading.Thread`` que ejecuta ``asyncio.run(coro_factory())``.

        Útil cuando el job debe sobrevivir al request actual o cuando los
        consumidores son síncronos (``BackgroundJobsService``).
        """

        def _target() -> None:
            try:
                asyncio.run(coro_factory())
            except Exception:  # noqa: BLE001 — defensivo, no propaga al main
                log.exception("Detached job %r raised", name)
            finally:
                with self._lock:
                    self._threads.pop(name, None)

        with self._lock:
            t = threading.Thread(target=_target, name=name, daemon=True)
            self._threads[name] = t
        t.start()

    # --- modo task (loop actual) -------------------------------------------

    def run_task(self, coro: Awaitable[Any], *, name: str) -> asyncio.Task:
        """Programa el coroutine como ``asyncio.create_task`` en el loop actual."""
        task = asyncio.create_task(coro, name=name)

        def _on_done(t: asyncio.Task) -> None:
            with self._lock:
                self._tasks.pop(name, None)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                log.exception("Task job %r raised", name, exc_info=exc)

        task.add_done_callback(_on_done)
        with self._lock:
            self._tasks[name] = task
        return task

    # --- shutdown ----------------------------------------------------------

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Best-effort: registra los jobs aún vivos al cerrar la app."""
        with self._lock:
            alive_threads = [t for t in self._threads.values() if t.is_alive()]
            alive_tasks = [t for t in self._tasks.values() if not t.done()]
        if alive_threads:
            log.info(
                "JobsScheduler shutdown: %d threads aún vivos: %s",
                len(alive_threads),
                [t.name for t in alive_threads],
            )
        if alive_tasks:
            log.info(
                "JobsScheduler shutdown: %d tasks aún vivas: %s",
                len(alive_tasks),
                [t.get_name() for t in alive_tasks],
            )
        # No forzamos join — los daemon threads mueren con el proceso y
        # las tasks se cancelan al cerrar el loop.

    # --- introspección para tests -----------------------------------------

    def active_thread_names(self) -> list[str]:
        with self._lock:
            return [t.name for t in self._threads.values() if t.is_alive()]

    def active_task_names(self) -> list[str]:
        with self._lock:
            return [t.get_name() for t in self._tasks.values() if not t.done()]
