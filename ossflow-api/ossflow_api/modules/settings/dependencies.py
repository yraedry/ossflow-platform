"""Dependencias FastAPI del módulo settings.

``_singleton`` mantiene un único ``SettingsService`` por proceso para que la
inicialización (BD, import legacy, migraciones) se ejecute una sola vez.
"""

from __future__ import annotations

from .service import SettingsService

_singleton: SettingsService | None = None


def get_settings_service() -> SettingsService:
    global _singleton
    if _singleton is None:
        _singleton = SettingsService()
    return _singleton


def reset_for_tests() -> None:
    """Limpia el singleton entre tests."""
    global _singleton
    _singleton = None
