"""Dependencias FastAPI del módulo scrapper."""

from __future__ import annotations

from ossflow_api.clients.scrapper import scrapper_client

from .service import ScrapperService


def _default_scan_cache_loader():
    """Devuelve el singleton ``LibraryCache`` del módulo library.

    Cierre acoplamiento #8 (T23.6): el scrapper recibe la **misma**
    instancia que usan los endpoints library, en lugar de instanciar
    un ``ScanCache`` paralelo (como hacía ``api/oracle.py``) o leer del
    global ``api.app._scan_cache`` (parche temporal de T24). Una sola
    instancia de cache → cero divergencia al escribir.
    """
    from ossflow_api.modules.library.dependencies import get_library_cache

    return get_library_cache()


def _default_patch_poster(cache, folder_name: str, saved: str) -> None:
    """Persiste ``poster_filename`` en la cache (canonical helper).

    Importa la función directamente del módulo library, no del shim
    legacy ``api.scan_cache``.
    """
    from ossflow_api.modules.library.cache import patch_poster_in_cache

    patch_poster_in_cache(cache, folder_name, saved)


def get_scrapper_service() -> ScrapperService:
    """Factory inyectada vía ``Depends()`` por el router."""
    from api.settings import get_library_path

    client = scrapper_client()
    return ScrapperService(
        splitter_url=client.base_url,
        library_path_loader=get_library_path,
        scan_cache_loader=_default_scan_cache_loader,
        patch_poster=_default_patch_poster,
    )
