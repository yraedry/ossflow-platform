"""Dependencias FastAPI del módulo logs."""

from __future__ import annotations

from .service import LogsService


def get_logs_service() -> LogsService:
    return LogsService()
