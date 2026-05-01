"""Dependencias FastAPI del módulo library.

Wiring DI:

* ``LibraryCache`` se construye **una sola vez** (singleton process-scope)
  apuntando a ``CONFIG_DIR/library.json``. Es el cierre del acoplamiento
  sucio #8 — todos los consumidores (library, scrapper) deben recibir
  esta misma instancia.
* ``library_path_loader`` resuelve la ruta de la biblioteca leyendo de
  ``api.settings`` con import diferido (evita import circular en startup).
* ``poster_downloader`` se inyecta desde ``scrapper.service`` para la
  feature de redownload — adapta la firma a ``Callable[..., Awaitable]``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .cache import LibraryCache
from .service import LibraryService

def _config_dir() -> Path:
    return Path(os.environ.get("CONFIG_DIR", "/config"))


@lru_cache(maxsize=1)
def _get_library_cache() -> LibraryCache:
    """Singleton ``LibraryCache`` apuntando a ``CONFIG_DIR/library.json``.

    Cierre del acoplamiento #8: todo el código (library, scrapper) usa
    esta instancia. ``api/app.py`` también la importa hasta T_LATE para
    sus endpoints aún no migrados.
    """
    return LibraryCache(_config_dir() / "library.json")


def get_library_cache() -> LibraryCache:
    """Factory público de ``LibraryCache`` (para inyección en otros módulos)."""
    return _get_library_cache()


def _default_library_path_loader() -> Optional[str]:
    """Loader de ``library_path`` con import diferido sobre ``api.settings``."""
    from api.settings import get_library_path

    return get_library_path()


async def _default_poster_downloader(
    folder: Path,
    poster_url: Optional[str],
    *,
    force: bool = False,
) -> Optional[str]:
    """Adapter del downloader del scrapper.

    Construye un ``ScrapperService`` puntual (es ligero — solo httpx +
    settings) y delega en ``download_poster``. Mantiene a ``library``
    desacoplado de la inicialización del scrapper.
    """
    from ossflow_api.modules.scrapper.dependencies import get_scrapper_service

    svc = get_scrapper_service()
    return await svc.download_poster(folder, poster_url, force=force)


@lru_cache(maxsize=1)
def _get_library_service() -> LibraryService:
    return LibraryService(
        cache=_get_library_cache(),
        library_path_loader=_default_library_path_loader,
        poster_downloader=_default_poster_downloader,
    )


def get_library_service() -> LibraryService:
    """Factory inyectada vía ``Depends()`` por el router.

    Singleton process-scope porque ``LibraryService`` mantiene el flag
    de coalescing ``_refresh_inflight`` — si fueran instancias
    distintas, un page-reload-storm encolaría N rescans paralelos.
    """
    return _get_library_service()
