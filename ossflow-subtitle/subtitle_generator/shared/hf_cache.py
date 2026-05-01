"""Helpers de la cache HuggingFace.

Migrado de ``app.py`` en T31.5. La cache local de modelos HF acumula
``.lock`` files cuando un worker muere a mitad de download. Este
módulo expone la utilidad para limpiarlos al startup del servicio
(o vía endpoint maintenance).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def hf_cache_root() -> Path:
    return Path(
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or "/models/huggingface"
    )


def clear_hf_locks() -> dict[str, Any]:
    """Borra ``.lock`` files antiguos dentro de la cache HF hub.

    Devuelve un summary con counts. Seguro: solo toca ficheros bajo
    ``<cache>/hub/.locks/`` (o legacy ``<cache>/.locks/``).
    """
    root = hf_cache_root()
    candidates = [root / "hub" / ".locks", root / ".locks"]
    removed, errors = 0, []
    for base in candidates:
        if not base.exists():
            continue
        for lock in base.rglob("*.lock"):
            try:
                lock.unlink()
                removed += 1
            except Exception as exc:
                errors.append(f"{lock}: {exc}")
    return {
        "removed": removed,
        "errors": errors,
        "roots": [str(c) for c in candidates],
    }
