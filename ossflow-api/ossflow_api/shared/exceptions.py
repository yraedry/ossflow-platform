"""Jerarquía de excepciones de dominio para ossflow-api.

Cada excepción declara un ``status_code`` HTTP por defecto. Los routers
las traducen a ``HTTPException`` o las dejan propagar a un exception
handler global.
"""

from __future__ import annotations


class ApiError(Exception):
    """Base de errores controlados por la API."""

    status_code: int = 500

    def __init__(self, message: str = "", *, status_code: int | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class NotFoundError(ApiError):
    """Recurso solicitado no existe."""

    status_code = 404


class ValidationError(ApiError):
    """Payload o estado inválido para la operación solicitada."""

    status_code = 400


class ConflictError(ApiError):
    """El recurso existe en un estado incompatible con la operación."""

    status_code = 409


class UpstreamError(ApiError):
    """Error al hablar con un microservicio backend."""

    status_code = 502
