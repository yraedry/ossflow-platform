"""Módulo dubbing: proxy hacia el backend dubbing-generator.

Expone los endpoints de listado de voces, transcript de muestras,
QA por video / por instructional, restart de mantenimiento y
``analyze`` (con o sin sintesis). La lógica vive en el microservicio
backend; aquí sólo traducimos paths host→container y delegamos
por HTTP.
"""

from .router import router as dubbing_router

__all__ = ["dubbing_router"]
