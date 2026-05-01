"""Factory del cliente para el microservicio splitter (chapter detection)."""

from __future__ import annotations

import os

from .base import BackendClient

_client: BackendClient | None = None


def splitter_client() -> BackendClient:
    """Devuelve un ``BackendClient`` cacheado para el splitter."""
    global _client
    if _client is None:
        _client = BackendClient(os.environ.get("SPLITTER_URL", "http://localhost:8001"))
    return _client


def reset() -> None:
    """Helper de tests: limpia la caché para que se relean env vars."""
    global _client
    _client = None
