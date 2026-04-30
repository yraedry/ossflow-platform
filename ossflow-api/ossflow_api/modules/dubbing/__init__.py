"""Módulo dubbing: proxy hacia el backend dubbing-generator + burn-subs.

Expone los endpoints de listado de voces, transcript de muestras,
QA por video / por instructional, restart de mantenimiento y
``analyze`` (con o sin sintesis). La lógica vive en el microservicio
backend; aquí sólo traducimos paths host→container y delegamos
por HTTP.

Adicionalmente absorbe ``/api/burn-subs`` (antiguo módulo separado, T22):
quema SRT en vídeo con ffmpeg como background job.
"""

from .burn_subs_router import router as burn_subs_router
from .router import router as dubbing_router

__all__ = ["dubbing_router", "burn_subs_router"]
