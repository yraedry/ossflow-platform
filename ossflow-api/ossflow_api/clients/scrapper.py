"""Factory del cliente para el microservicio scrapper (oracle backend).

Apunta a ``SPLITTER_URL`` por defecto: el subsistema oracle vive dentro
del contenedor ``chapter-splitter`` y comparte URL. Mantenemos una env
var dedicada (``SCRAPPER_URL``) para permitir desacoplarlos en el
futuro sin cambios en código.
"""

from __future__ import annotations

import os

from .base import BackendClient

_client: BackendClient | None = None


def scrapper_client() -> BackendClient:
    """Devuelve un ``BackendClient`` cacheado para el scrapper/oracle."""
    global _client
    if _client is None:
        # ``SCRAPPER_URL`` tiene preferencia para futuras separaciones;
        # cae a ``SPLITTER_URL`` para preservar el comportamiento actual.
        base = (
            os.environ.get("SCRAPPER_URL")
            or os.environ.get("SPLITTER_URL")
            or "http://chapter-splitter:8001"
        )
        _client = BackendClient(base)
    return _client


def reset() -> None:
    """Helper de tests: limpia la caché para que se relean env vars."""
    global _client
    _client = None
