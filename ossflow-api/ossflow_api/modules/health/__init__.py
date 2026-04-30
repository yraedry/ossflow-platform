"""Módulo health: agrega los /health de los microservicios backend."""

from .router import router as health_router

__all__ = ["health_router"]
