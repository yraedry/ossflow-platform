"""Módulo scrapper: proxy/orquestador del subsistema oracle del backend.

Renombrado completo (Plan 2 T24 + T_LATE_1): paquete Python, prefix HTTP
(``/api/scrapper``), tags OpenAPI, frontend feature. El subsistema del
microservicio chapter-splitter sigue exponiéndose internamente como
``/oracle/*`` (no es API pública).
"""

from .router import router as scrapper_router

__all__ = ["scrapper_router"]
