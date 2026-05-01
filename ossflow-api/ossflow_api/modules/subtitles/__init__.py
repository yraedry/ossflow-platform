"""Módulo subtitles: proxy hacia el backend subtitle-generator.

Expone endpoints de validación, regeneración por segmento, traducción y
análisis de vídeo. La lógica vive en el microservicio backend; aquí
sólo traducimos paths host→container y delegamos por HTTP.
"""

from .router import router as subtitles_router

__all__ = ["subtitles_router"]
