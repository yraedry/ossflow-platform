"""Servicio de browsing de filesystem para el library picker.

Encapsula la lógica de los endpoints ``/api/fs/browse`` y ``/api/browse``.
Son operaciones de directorio puro (listar carpetas + vídeos) que se
usan para que el frontend permita al usuario navegar el árbol antes
de configurar ``library_path``.

Diferencias entre ambos endpoints:

* ``fs_browse`` — clamp duro a ``MEDIA_ROOT``. Anti-traversal estricto.
* ``browse`` — sin clamp. Default a ``library_path``, fallback a
  ``MEDIA_ROOT``. El usuario ya tiene acceso al filesystem vía NAS.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}


class _OutOfRoot(Exception):
    """fs_browse: el path resuelto está fuera de MEDIA_ROOT."""


class _NoMediaRoot(Exception):
    """fs_browse: ``MEDIA_ROOT`` no existe en disco."""


class _NotADir(Exception):
    """El path resuelto no es un directorio."""


class _NoPermission(Exception):
    """No se puede leer el directorio (PermissionError)."""


def _media_root() -> Path:
    """Resuelve ``MEDIA_ROOT`` desde el entorno (re-evaluado por test)."""
    return Path(os.environ.get("MEDIA_ROOT", "/media"))


def fs_browse(path: str = "") -> dict[str, Any]:
    """Lista subdirectorios bajo ``MEDIA_ROOT``. Anti-traversal estricto."""
    root = _media_root().resolve()
    if not root.exists():
        raise _NoMediaRoot(f"MEDIA_ROOT no accesible: {root}")

    target = (root if not path else Path(path)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise _OutOfRoot("Path fuera de MEDIA_ROOT") from exc

    if not target.exists() or not target.is_dir():
        raise _NotADir("Directorio no existe")

    entries = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            if not child.is_dir():
                continue
            entries.append({"name": child.name, "path": str(child)})
    except PermissionError as exc:
        raise _NoPermission("Sin permisos de lectura") from exc

    parent = None if target == root else str(target.parent)
    return {
        "root": str(root),
        "path": str(target),
        "parent": parent,
        "entries": entries,
    }


def browse(
    path: Optional[str],
    library_path_loader: Callable[[], Optional[str]],
) -> dict[str, Any]:
    """Browse libre: directorios + vídeos.

    Default: ``library_path`` si está configurado y existe, si no
    ``MEDIA_ROOT``. Devuelve ``current``, ``parent``, ``directories``,
    ``files``.
    """
    library_path = library_path_loader()
    media_root = _media_root()

    if not path:
        if library_path and Path(library_path).exists():
            target = Path(library_path)
        else:
            target = media_root
    else:
        target = Path(path)

    try:
        target = target.resolve()
    except OSError:
        pass

    if not target.exists():
        if media_root.exists():
            target = media_root
        else:
            raise FileNotFoundError(f"Ruta no encontrada: {target}")

    if not target.is_dir():
        raise _NotADir(f"No es un directorio: {target}")

    parent_path = target.parent
    parent = None if parent_path == target else str(parent_path)

    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    try:
        entries = list(target.iterdir())
    except PermissionError as exc:
        raise _NoPermission(f"Sin permisos para leer: {target}") from exc

    for entry in entries:
        try:
            if entry.is_dir():
                directories.append({"name": entry.name, "path": str(entry)})
            elif entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
                files.append({
                    "name": entry.name,
                    "path": str(entry),
                    "size": entry.stat().st_size,
                })
        except (PermissionError, OSError):
            continue

    directories.sort(key=lambda d: d["name"].lower())
    files.sort(key=lambda f: f["name"].lower())

    return {
        "current": str(target),
        "parent": parent,
        "directories": directories,
        "files": files,
    }
