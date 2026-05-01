"""Dependencias FastAPI del módulo preflight.

``_singleton`` mantiene un único ``PreflightService`` por proceso para que
la cache + locks por path se compartan. ``infrastructure.lifespan`` registra
``PreflightService.aclose`` como hook de shutdown (rotura acoplamiento #5).
"""

from __future__ import annotations

from .service import PreflightService

_singleton: PreflightService | None = None


def get_preflight_service() -> PreflightService:
    global _singleton
    if _singleton is None:
        _singleton = PreflightService()
    return _singleton


def reset_for_tests() -> None:
    global _singleton
    _singleton = None
