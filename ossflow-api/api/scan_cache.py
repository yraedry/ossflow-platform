"""Compat shim. Lógica movida a ``ossflow_api.modules.library.cache``.

Mantiene la API legacy (``ScanCache``, ``find_poster``,
``find_poster_cached``, ``enrich_with_poster``, ``patch_poster_in_cache``,
``POSTER_NAMES``) para que ``app.py``, ``pipeline.py``, ``library_refresh.py``,
y los módulos no migrados sigan funcionando hasta que migren a importar
desde ``ossflow_api.modules.library``.

``ScanCache`` se renombró internamente a ``LibraryCache`` durante T23.1.
Aquí se reexporta como ``ScanCache`` para retrocompat.
"""

from ossflow_api.modules.library.cache import (  # noqa: F401
    POSTER_NAMES,
    LibraryCache as ScanCache,
    enrich_with_poster,
    find_poster,
    find_poster_cached,
    patch_poster_in_cache,
)
