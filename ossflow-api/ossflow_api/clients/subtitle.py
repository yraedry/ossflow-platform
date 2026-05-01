"""Factory del cliente para el microservicio subtitle (WhisperX)."""

from __future__ import annotations

import os

from .base import BackendClient

_client: BackendClient | None = None


def subs_client() -> BackendClient:
    """Devuelve un ``BackendClient`` cacheado para subtitle-generator."""
    global _client
    if _client is None:
        _client = BackendClient(os.environ.get("SUBS_URL", "http://localhost:8002"))
    return _client


def reset() -> None:
    """Helper de tests: limpia la caché para que se relean env vars."""
    global _client
    _client = None
