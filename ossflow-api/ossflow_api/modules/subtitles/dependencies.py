"""Dependencias FastAPI del módulo subtitles."""

from __future__ import annotations

from ossflow_api.clients.subtitle import subs_client

from .service import SubtitlesService


def get_subtitles_service() -> SubtitlesService:
    """Factory inyectada vía ``Depends()`` por el router.

    ``api.settings.get_library_path`` y ``api.settings.get_setting`` siguen
    siendo la fuente única de verdad mientras settings no se haya migrado
    al patrón vertical slice por completo. Import diferido para no
    acoplar el módulo a settings en tiempo de import.
    """
    from api.settings import get_library_path, get_setting

    client = subs_client()
    return SubtitlesService(
        library_path=get_library_path(),
        subs_url=client.base_url,
        setting_getter=get_setting,
    )
