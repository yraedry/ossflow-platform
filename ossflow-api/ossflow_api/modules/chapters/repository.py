"""Acceso a filesystem del módulo chapters.

Encapsula `os.rename` y la búsqueda de hermanos (sidecars) para que el
``ChaptersService`` pueda razonar sobre paths sin depender de I/O directa.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .schemas import SIDECAR_SUFFIXES

log = logging.getLogger(__name__)


class ChaptersRepository:
    """Operaciones sobre archivos: renombre y búsqueda de sidecars."""

    @staticmethod
    def rename(source: Path, dest: Path) -> None:
        """Renombra ``source`` → ``dest`` (sin sobreescribir; el service valida)."""
        os.rename(source, dest)

    @staticmethod
    def iter_dir(path: Path) -> list[Path]:
        """Lista ordenada de entradas de ``path`` (no recursivo)."""
        return sorted(path.iterdir())

    @staticmethod
    def find_sibling(stem_path: Path, suffix: str) -> Path | None:
        """Devuelve el path hermano con ``suffix`` reemplazando la ext, o None."""
        candidate = stem_path.with_name(stem_path.stem + suffix)
        return candidate if candidate.exists() else None

    def rename_siblings(
        self,
        anchor_dir_path: Path,
        old_stem: str,
        new_stem: str,
    ) -> list[dict[str, str]]:
        """Renombra todos los sidecars conocidos manteniendo el sufijo.

        ``anchor_dir_path`` es cualquier path dentro del directorio destino;
        se usa con ``with_name`` para construir las rutas hermanas. Devuelve
        la lista ``[{"from": ..., "to": ...}]`` con los renombres efectivos.
        """
        renamed: list[dict[str, str]] = []
        for suffix in SIDECAR_SUFFIXES:
            sib_old = anchor_dir_path.with_name(old_stem + suffix)
            if not sib_old.exists():
                continue
            sib_new = anchor_dir_path.with_name(new_stem + suffix)
            if sib_new == sib_old:
                continue
            if sib_new.exists():
                log.warning("Sidecar target already exists, skipping: %s", sib_new)
                continue
            self.rename(sib_old, sib_new)
            renamed.append({"from": str(sib_old), "to": str(sib_new)})
        return renamed
