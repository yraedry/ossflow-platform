"""Dependencias FastAPI para el módulo health."""

from __future__ import annotations

from .service import HealthService


def get_health_service() -> HealthService:
    """Devuelve un ``HealthService`` listo para inyectar en el router.

    Lo construimos en cada request: el servicio es stateless y los
    ``httpx.AsyncClient`` viven dentro de cada llamada.
    """
    return HealthService()
