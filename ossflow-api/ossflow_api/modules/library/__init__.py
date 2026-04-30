"""Módulo library: gestión de la biblioteca de instructionals.

Anatomía complejo (spec base §3.4):
* ``cache.py`` — ``LibraryCache`` singleton de la cache JSON. Sustituye
  al antiguo ``api/scan_cache.py:ScanCache`` y al duplicado que tenía
  ``oracle.py``. Cierre del acoplamiento sucio #8.
* ``refresh.py`` — utilidades de re-stat ligero por instructional.
* ``service.py`` — ``LibraryService`` con scan, browse, video-info,
  thumbnail, media, poster.
* ``router.py`` — endpoints HTTP (``/api/scan``, ``/api/library*``,
  ``/api/fs/*``, ``/api/mount``, ``/api/browse``, ``/api/video-info``,
  ``/api/thumbnail``, ``/api/media``).
* ``dependencies.py`` — wiring DI (singletons scope-app).
* ``schemas.py`` — DTOs Pydantic.
"""

from .cache import (
    LibraryCache,
    POSTER_NAMES,
    enrich_with_poster,
    find_poster,
    find_poster_cached,
    patch_poster_in_cache,
)
from .dependencies import get_library_cache, get_library_service
from .router import router as library_router
from .service import LibraryService

__all__ = [
    "LibraryCache",
    "LibraryService",
    "POSTER_NAMES",
    "enrich_with_poster",
    "find_poster",
    "find_poster_cached",
    "get_library_cache",
    "get_library_service",
    "library_router",
    "patch_poster_in_cache",
]
