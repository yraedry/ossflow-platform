"""Endpoints HTTP del módulo health."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from .dependencies import get_health_service
from .service import HealthService

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/backends")
async def backends_health(
    service: HealthService = Depends(get_health_service),
) -> dict:
    """Devuelve el estado de todos los backends conocidos en paralelo."""
    return await service.ping_all()


@router.get("/{service_name}")
async def one_backend(
    service_name: str,
    service: HealthService = Depends(get_health_service),
) -> dict:
    """Devuelve el estado de un backend concreto.

    404 si el nombre no está en el mapa de backends.
    """
    if not service.is_known(service_name):
        raise HTTPException(status_code=404, detail=f"unknown service '{service_name}'")
    return await service.ping_one(service_name)
