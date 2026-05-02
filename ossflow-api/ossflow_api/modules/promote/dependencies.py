"""Dependencias FastAPI del módulo promote.

``_singleton`` mantiene un único ``PromoteService`` por proceso para que
los locks por season se compartan entre requests (el patrón legacy era un
``dict`` global de módulo; ahora vive como atributo de instancia del
singleton).
"""

from __future__ import annotations

from .service import PromoteService

_singleton: PromoteService | None = None


def _default_cache_factory() -> object:
    """Devuelve el singleton ``LibraryCache`` del módulo library.

    Cierre acoplamiento #8 (T23.6): un único cache compartido por todos
    los módulos. Antes promote instanciaba un ``ScanCache`` paralelo que
    leía el mismo fichero pero divergía en runtime de la cache de
    library/scrapper.
    """
    from ossflow_api.modules.library.dependencies import get_library_cache

    return get_library_cache()


def _default_refresh_flags(item: dict) -> None:
    """Wrapper diferido a ``modules.library.refresh.rediscover_instructional``.

    Promote borra el ``.mp4`` original y crea un ``.mkv`` nuevo. Por eso
    necesitamos rediscover (re-walk filesystem) en vez de refresh_flags
    (solo re-stat de entradas existentes), que no descubriría el .mkv
    recién creado y dejaría la cache mostrando un instructional sin el
    nuevo capítulo promovido — el botón "Promover" seguiría apareciendo.
    """
    from ossflow_api.modules.library.refresh import rediscover_instructional

    rediscover_instructional(item)


def _default_library_path_loader() -> str | None:
    """Wrapper diferido a ``api.settings.get_library_path``."""
    from api.settings import get_library_path

    return get_library_path()


def get_promote_service() -> PromoteService:
    """Factory inyectada vía ``Depends()`` por el router.

    Devuelve un singleton scope-app para que los locks por season vivan
    durante toda la vida del proceso (de otro modo dos requests
    concurrentes para la misma season verían instancias distintas y la
    serialización se perdería).
    """
    global _singleton
    if _singleton is None:
        _singleton = PromoteService(
            library_path_loader=_default_library_path_loader,
            cache_factory=_default_cache_factory,
            refresh_flags=_default_refresh_flags,
        )
    return _singleton


def reset_for_tests() -> None:
    """Limpia el singleton entre tests."""
    global _singleton
    _singleton = None
