"""Módulo settings: persistencia de configuración dinámica en SQLite."""

from .router import router as settings_router
from .service import SettingsService
from .schemas import DEFAULTS, SECRET_KEYS, CONFIG_DIR

__all__ = [
    "settings_router",
    "SettingsService",
    "DEFAULTS",
    "SECRET_KEYS",
    "CONFIG_DIR",
]
