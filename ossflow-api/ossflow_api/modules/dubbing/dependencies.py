"""Dependencias FastAPI del módulo dubbing."""

from __future__ import annotations

from typing import Optional

from ossflow_api.clients.dubbing import dubbing_client
from ossflow_api.shared.voice_profiles import load_voice_profile_for_path

from .service import DubbingService


def _default_scan_cache_loader() -> Optional[dict]:
    """Carga la cache de escaneo de la librería desde disco.

    Import diferido para no acoplar el módulo a ``modules.library`` ni a
    ``api.settings`` en tiempo de import (mismo patrón que el resto de
    módulos migrados). Usa el singleton ``LibraryCache`` del módulo
    library — cierre acoplamiento #8 (T23.6).
    """
    from ossflow_api.modules.library.dependencies import get_library_cache

    return get_library_cache().load()


def get_dubbing_service() -> DubbingService:
    """Factory inyectada vía ``Depends()`` por el router.

    ``api.settings.get_library_path`` sigue siendo la fuente única de
    verdad mientras settings no se haya migrado al patrón vertical
    slice por completo. Import diferido para no acoplar el módulo a
    settings en tiempo de import.
    """
    from api.settings import get_library_path

    client = dubbing_client()
    return DubbingService(
        library_path=get_library_path(),
        dubbing_url=client.base_url,
        voice_profile_loader=load_voice_profile_for_path,
        scan_cache_loader=_default_scan_cache_loader,
    )
