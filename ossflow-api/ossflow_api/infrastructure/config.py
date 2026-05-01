"""Configuración global por entorno.

Centraliza la lectura de variables de entorno y rutas constantes para
sustituir progresivamente los `os.environ.get(...)` dispersos por el código.
"""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_DIR: Path = Path(os.environ.get("CONFIG_DIR", "/data/config"))
"""Directorio donde viven los ficheros legacy JSON y los .bak de migración."""

MEDIA_ROOT: Path = Path(os.environ.get("MEDIA_ROOT", "/media"))
"""Raíz de la biblioteca dentro del contenedor (montaje CIFS/NAS)."""

LIBRARY_DEFAULT_PATH: str = os.environ.get("LIBRARY_PATH", "")
"""Path por defecto de la biblioteca cuando no hay setting persistido."""

DB_PATH: str = os.environ.get(
    "OSSFLOW_DB_PATH",
    os.environ.get("BJJ_DB_PATH", "/data/db/bjj.db"),
)
"""Ruta de la BD SQLite. ``OSSFLOW_DB_PATH`` es canónica; ``BJJ_DB_PATH`` se
mantiene como fallback durante la transición del split de repos."""
