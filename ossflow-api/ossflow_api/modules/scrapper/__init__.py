"""Módulo scrapper: proxy/orquestador del subsistema scrapper del backend.

Renombrado completo (Plan 2 T24 + T_LATE_1 + T_LATE_3): paquete Python,
prefix HTTP (``/api/scrapper``), tags OpenAPI, frontend feature, y
endpoints internos del microservicio chapter-splitter (``/scrapper/*``).
"""

from .router import router as scrapper_router

__all__ = ["scrapper_router"]
