"""Factory del cliente para el microservicio telegram-fetcher."""

from __future__ import annotations

import os

from .base import BackendClient

_client: BackendClient | None = None


def telegram_client() -> BackendClient:
    """Devuelve un ``BackendClient`` cacheado para telegram-fetcher.

    El backend monta sus routers bajo el prefijo ``/telegram``; ese segmento
    se compone en ``TelegramService`` para no repetirlo en cada llamada.
    """
    global _client
    if _client is None:
        _client = BackendClient(
            os.environ.get("TELEGRAM_FETCHER_URL", "http://telegram-fetcher:8004")
        )
    return _client


def reset() -> None:
    """Helper de tests: limpia la caché para que se relean env vars."""
    global _client
    _client = None
