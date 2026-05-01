"""Orquestador de hooks de lifespan.

Cada módulo registra sus callables de startup/shutdown vía
``register_startup`` / ``register_shutdown``. ``main.py`` enchufa el
``lifespan`` resultante en ``FastAPI(lifespan=...)``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, List, Union

from fastapi import FastAPI

log = logging.getLogger(__name__)

Hook = Callable[[], Union[None, Awaitable[None]]]

_startup_hooks: List[Hook] = []
_shutdown_hooks: List[Hook] = []


def register_startup(hook: Hook) -> None:
    """Registra un hook que se ejecuta al arrancar la app.

    Acepta funciones síncronas o asíncronas.
    """
    _startup_hooks.append(hook)


def register_shutdown(hook: Hook) -> None:
    """Registra un hook que se ejecuta al cerrar la app."""
    _shutdown_hooks.append(hook)


async def _run_hook(hook: Hook) -> None:
    result = hook()
    if hasattr(result, "__await__"):
        await result  # type: ignore[func-returns-value]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Context manager que FastAPI invoca al arrancar y al cerrar."""
    for hook in _startup_hooks:
        try:
            await _run_hook(hook)
        except Exception:
            log.exception("Startup hook %r falló", hook)
    yield
    for hook in _shutdown_hooks:
        try:
            await _run_hook(hook)
        except Exception:
            log.exception("Shutdown hook %r falló", hook)
