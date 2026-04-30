"""Endpoints HTTP del módulo metrics."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from .dependencies import get_metrics_service
from .service import MetricsService

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/")
async def get_metrics(svc: MetricsService = Depends(get_metrics_service)) -> dict:
    """Devuelve un snapshot CPU/RAM/Disco/GPU (cacheado 5s)."""
    return await svc.snapshot()
