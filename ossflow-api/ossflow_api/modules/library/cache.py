"""Cache persistente del último resultado de ``scan_library``.

Migrado desde ``api/scan_cache.py``. Mismo contrato — solo renombrado:
``ScanCache`` → ``LibraryCache``. Funciones libres (``find_poster``,
``find_poster_cached``, ``enrich_with_poster``, ``patch_poster_in_cache``)
mantienen su firma original para no romper consumidores externos.

Concurrencia: lectura/escritura protegidas por ``threading.Lock`` y
escritura atómica vía ``os.replace`` — sin esto un lector podía leer
``{}`` mientras otra request terminaba de escribir, y dos escrituras
concurrentes dejaban la más vieja en disco.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

POSTER_NAMES = (
    "poster.jpg", "poster.png", "poster.webp",
    "cover.jpg", "cover.png", "cover.webp",
    "folder.jpg",
)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


class LibraryCache:
    """Persiste el resultado más reciente de ``scan_library`` como JSON.

    Renombrada de ``ScanCache`` (T23.1) para alinear con el nombre del
    módulo. La API pública es idéntica: ``exists``/``load``/``save``.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Optional[dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return None
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Failed to load library cache %s: %s", self.path, exc)
                return None

    def save(self, instructionals: list[dict[str, Any]]) -> None:
        with self._lock:
            self._ensure_dir()
            payload = {"instructionals": instructionals}
            text = json.dumps(payload, indent=2, ensure_ascii=False)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# Funciones libres — mantienen la firma del antiguo ``api/scan_cache.py`` para
# no romper consumidores externos durante la migración. Internamente el
# módulo nuevo las puede usar; el shim de compat reexporta.
# ---------------------------------------------------------------------------


def find_poster(folder: Path) -> Optional[Path]:
    """Busca un fichero de poster en ``folder`` (case-insensitive).

    Estrategia: prefiere los nombres canónicos (poster/cover/folder),
    luego cualquier imagen del root (ordenada para determinismo).
    Devuelve ``None`` si no hay nada o no se puede leer la carpeta.
    """
    if not folder.exists() or not folder.is_dir():
        return None
    lowered = {n.lower(): n for n in POSTER_NAMES}
    try:
        entries = list(folder.iterdir())
    except (PermissionError, OSError):
        return None
    fallback: list[Path] = []
    for entry in entries:
        if not entry.is_file():
            continue
        name_lower = entry.name.lower()
        if name_lower in lowered:
            return entry
        if name_lower.endswith(_IMAGE_EXTS):
            fallback.append(entry)
    if fallback:
        return sorted(fallback, key=lambda p: p.name.lower())[0]
    return None


def find_poster_cached(folder: Path, poster_filename: Optional[str]) -> Optional[Path]:
    """Resuelve poster prefiriendo el filename cacheado para evitar ``iterdir`` en NAS.

    Si ``poster_filename`` se provee y ``folder / poster_filename`` existe,
    lo devuelve directo. Si no, fallback a :func:`find_poster` (con iterdir).
    """
    if poster_filename:
        try:
            candidate = folder / poster_filename
            if candidate.is_file():
                return candidate
        except OSError:
            pass
    return find_poster(folder)


def enrich_with_poster(instructionals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Añade ``has_poster`` + ``poster_filename`` + ``poster_mtime`` a cada item."""
    for item in instructionals:
        folder = item.get("path")
        poster = find_poster(Path(folder)) if folder else None
        item["has_poster"] = poster is not None
        item["poster_filename"] = poster.name if poster else None
        if poster is not None:
            try:
                item["poster_mtime"] = int(poster.stat().st_mtime)
            except OSError:
                item["poster_mtime"] = None
        else:
            item["poster_mtime"] = None
    return instructionals


def patch_poster_in_cache(
    cache: LibraryCache,
    instructional_name: str,
    poster_filename: Optional[str],
) -> bool:
    """Actualiza ``has_poster``/``poster_filename`` para un instructional.

    Devuelve ``True`` si la cache se actualizó. Lo usan los flujos
    ad-hoc de poster (auto-download de scrapper, upload manual) para que
    el endpoint ``/api/library`` refleje el nuevo poster sin un rescan
    completo.
    """
    data = cache.load()
    if not data or not isinstance(data, dict):
        return False
    items = data.get("instructionals")
    if not isinstance(items, list):
        return False
    changed = False
    for item in items:
        if item.get("name") == instructional_name:
            item["has_poster"] = bool(poster_filename)
            item["poster_filename"] = poster_filename
            mtime_val: Optional[int] = None
            folder = item.get("path")
            if poster_filename and folder:
                try:
                    mtime_val = int((Path(folder) / poster_filename).stat().st_mtime)
                except OSError:
                    mtime_val = None
            item["poster_mtime"] = mtime_val
            changed = True
            break
    if changed:
        cache.save(items)
    return changed
