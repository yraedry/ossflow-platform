"""Repositorio de cleanup: acceso al filesystem.

Solo IO crudo: ``os.walk``, ``stat``, ``unlink``, ``rmdir``. Sin lógica
de negocio (clasificación de qué borrar) — eso vive en el servicio.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mkv", ".mp4"}
TEMP_EXTS = {".tmp", ".part", ".crdownload", ".bak"}


def safe_stat(p: Path) -> tuple[int, float] | None:
    try:
        st = p.stat()
        return st.st_size, st.st_mtime
    except OSError:
        return None


def info(p: Path) -> dict[str, Any] | None:
    s = safe_stat(p)
    if s is None:
        return None
    size, mtime = s
    return {"path": str(p), "size": int(size), "mtime": float(mtime)}


def is_temp_file(name: str) -> bool:
    lower = name.lower()
    if lower.startswith("~"):
        return True
    ext = os.path.splitext(lower)[1]
    return ext in TEMP_EXTS


class CleanupRepository:
    """Acceso al filesystem para escanear y borrar."""

    def walk(self, root: Path) -> Iterator[tuple[str, list[str], list[str]]]:
        """``os.walk`` envuelto para poder mockear en tests."""
        return os.walk(root)

    def delete_file(self, path: Path) -> int:
        """Borra un fichero y devuelve los bytes liberados."""
        st = safe_stat(path)
        size = st[0] if st else 0
        path.unlink()
        return int(size)

    def delete_empty_dir(self, path: Path) -> None:
        path.rmdir()

    def is_dir_empty(self, path: Path) -> bool:
        return not any(path.iterdir())
