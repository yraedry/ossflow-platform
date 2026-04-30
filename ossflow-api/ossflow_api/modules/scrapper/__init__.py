"""Módulo scrapper: proxy/orquestador del subsistema oracle.

Renombrado interno de ``oracle`` → ``scrapper`` (Plan 2 T24). El prefix
HTTP se mantiene como ``/api/oracle`` para no romper el frontend.
"""

from .router import router as scrapper_router

__all__ = ["scrapper_router"]
