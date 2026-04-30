"""Dependencias FastAPI del módulo chapters."""

from __future__ import annotations

from .service import ChaptersService


def get_chapters_service() -> ChaptersService:
    """Factory inyectada vía ``Depends()`` por el router.

    ``api.settings.get_library_path`` sigue siendo la fuente única de verdad
    mientras settings no se haya migrado. Cuando se migre, este import
    cambiará. Import diferido para no acoplar el módulo a settings al
    importarse desde tests u otros lugares.
    """
    from api.settings import get_library_path

    return ChaptersService(get_library_path())
