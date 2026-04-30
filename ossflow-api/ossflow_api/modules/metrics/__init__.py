"""Módulo metrics: snapshot CPU/RAM/Disco/GPU."""

from .router import router as metrics_router
from .service import MetricsService

__all__ = ["metrics_router", "MetricsService"]
