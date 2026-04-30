"""Módulo telegram: proxy hacia el backend telegram-fetcher.

Expone los endpoints de status, autenticación, gestión de canales,
sincronización (con SSE), listado/edición de media y descarga (con
SSE). La lógica vive en el microservicio backend; aquí sólo
validamos inputs, traducimos a llamadas HTTP y, opcionalmente,
registramos un job en el dashboard del processor-api.
"""

from .router import router as telegram_router

__all__ = ["telegram_router"]
