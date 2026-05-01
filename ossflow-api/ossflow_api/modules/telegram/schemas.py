"""DTOs y constantes del módulo telegram.

El proxy histórico hacia ``telegram-fetcher`` no usaba Pydantic — la
validación se hacía a mano sobre dicts crudos. Conservamos ese
contrato para no romper consumidores: este módulo expone constantes
de timeout y utilidades de validación, no modelos completos.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException

# Timeouts: idénticos a los del antiguo ``api/telegram.py`` para no
# alterar el comportamiento percibido por el frontend.
DEFAULT_TIMEOUT = 30.0
SSE_TIMEOUT = httpx.Timeout(None, connect=10.0)  # streams long-lived


def require_str(body: dict[str, Any], key: str) -> str:
    """Devuelve ``body[key]`` si es un string no vacío; si no, levanta 422."""
    val = body.get(key)
    if not isinstance(val, str) or not val.strip():
        raise HTTPException(
            status_code=422, detail=f"'{key}' must be a non-empty string"
        )
    return val
