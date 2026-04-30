"""Compat shim. Lógica movida a ``ossflow_api.modules.settings``.

Mantiene la API pública legacy (``load_settings``, ``save_settings``,
``get_library_path``, ``get_setting``, ``CONFIG_DIR``, ``router``) hasta
que todos los módulos consumidores migren al patrón Vertical Slice.

Compat con tests legacy: estos tests hacen ``importlib.reload(api.settings)``
después de mover ``CONFIG_DIR`` / ``BJJ_DB_PATH`` para reinicializar la BD
con el nuevo path. Para preservar ese contrato, al reimportar este shim
recargamos los módulos del Vertical Slice y reseteamos su singleton.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

# Recargar schemas/service para que recojan env vars (CONFIG_DIR,
# BJJ_DB_PATH) actualizadas en el test antes del reload del shim.
from ossflow_api.modules.settings import schemas as _schemas
from ossflow_api.modules.settings import service as _service

importlib.reload(_schemas)
importlib.reload(_service)

from ossflow_api.modules.settings.dependencies import (  # noqa: E402
    get_settings_service,
    reset_for_tests as _reset_singleton,
)
from ossflow_api.modules.settings.router import router  # noqa: F401, E402
from ossflow_api.modules.settings.schemas import CONFIG_DIR  # noqa: F401, E402

# Forzar reinstanciación del SettingsService para que la próxima llamada
# pase por ``ensure_initialized`` con el engine recién reseteado.
_reset_singleton()


def load_settings() -> dict[str, Any]:
    return get_settings_service().load()


def save_settings(data: dict[str, Any]) -> None:
    get_settings_service().save(data)


def get_library_path() -> Optional[str]:
    return get_settings_service().get_library_path()


def get_setting(key: str) -> Any:
    return get_settings_service().get(key)
