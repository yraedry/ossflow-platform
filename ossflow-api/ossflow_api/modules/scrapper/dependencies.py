"""Dependencias FastAPI del módulo scrapper."""

from __future__ import annotations

from ossflow_api.clients.scrapper import scrapper_client

from .service import ScrapperService


def _default_scan_cache_loader():
    """Devuelve la instancia global ``_scan_cache`` de ``api.app``.

    DEUDA TÉCNICA (acoplamiento #8 — cierre parcial T24):

    El servicio recibe la instancia de ``ScanCache`` por DI en lugar de
    crear su propia copia (como hacía ``api/oracle.py``). Eso elimina la
    divergencia entre ambas instancias. Sin embargo, sigue habiendo un
    import diferido al global ``api.app._scan_cache`` porque todavía no
    existe el módulo ``library`` (T23). Cuando T23 se complete, este
    loader desaparece y el módulo se inyectará con un repositorio del
    módulo ``library``.
    """
    from api.app import _scan_cache

    return _scan_cache


def _default_patch_poster(cache, folder_name: str, saved: str) -> None:
    """Wrap de ``api.scan_cache.patch_poster_in_cache``.

    Import diferido — el módulo ``api.scan_cache`` se migrará en T23.
    """
    from api.scan_cache import patch_poster_in_cache

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
