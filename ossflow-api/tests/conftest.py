"""Shared pytest config for processor-api tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure processor-api root is on sys.path so `import api.*` works.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default CONFIG_DIR to a scratch dir so module imports don't touch /data/config
os.environ.setdefault("CONFIG_DIR", str(ROOT / ".pytest_config"))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_library_cache_singleton():
    """T23.3: tests legacy hacen ``importlib.reload(api.app)`` para apuntar
    a un ``CONFIG_DIR`` distinto. Como ``LibraryCache`` ahora es un
    singleton process-scope (lru_cache) en ``modules/library/dependencies``,
    el reload no recrea la instancia y los tests escriben/leen al
    ``CONFIG_DIR`` del primer test que la creó. Limpiamos el cache antes
    de cada test para que cada uno reciba una instancia nueva con su
    propio path.
    """
    try:
        from ossflow_api.modules.library.dependencies import (
            _get_library_cache,
            _get_library_service,
        )
        _get_library_cache.cache_clear()
        _get_library_service.cache_clear()
    except Exception:
        pass
    yield
