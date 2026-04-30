"""Dependencias FastAPI del módulo metrics.

``_singleton`` mantiene un único ``MetricsService`` por proceso para que la
TTL cache y el ``httpx.AsyncClient`` se compartan entre requests.
``infrastructure.lifespan`` registra ``aclose`` en shutdown.
"""

from __future__ import annotations

from .service import MetricsService

_singleton: MetricsService | None = None


def get_metrics_service() -> MetricsService:
    global _singleton
    if _singleton is None:
        # Import diferido: settings.load_settings es la fuente de verdad
        # actual; cuando settings se migre, este import cambiará.
        from api.settings import load_settings
        _singleton = MetricsService(load_settings=load_settings)
    return _singleton


def reset_for_tests() -> None:
    """Limpia el singleton entre tests."""
    global _singleton
    _singleton = None
