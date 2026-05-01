"""Servicio de cleanup: clasifica artefactos y orquesta borrado.

Mantiene la lógica idéntica al antiguo ``api/cleanup.py``:
* ``orphan_srt``: subtítulos sin vídeo hermano
* ``old_dubbed``: ``*_DOBLADO.{mkv,mp4}`` más antiguos que el SRT hermano
* ``temp_files``: ``.tmp``, ``.part``, ``.crdownload``, ``~*``, ``*.bak``
* ``empty_dirs``: directorios vacíos

El borrado revalida cada path contra ``library_path`` (anti traversal) y
nunca borra vídeos que no sean ``*_DOBLADO.{mkv,mp4}``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from ossflow_api.modules.jobs.services.background import BackgroundJobsService

from .repository import (
    CleanupRepository,
    VIDEO_EXTS,
    info,
    is_temp_file,
    safe_stat,
)

log = logging.getLogger(__name__)


class CleanupService:
    """Orquesta el escaneo y borrado de artefactos."""

    def __init__(
        self,
        repo: CleanupRepository,
        jobs: BackgroundJobsService,
        library_path_loader: Callable[[], Optional[str]],
    ) -> None:
        self._repo = repo
        self._jobs = jobs
        self._library_path_loader = library_path_loader

    # --- validación de paths ---------------------------------------------

    def resolve_under_library(self, raw_path: str) -> Path:
        """Valida que ``raw_path`` esté bajo ``library_path``.

        Lanza ``ValueError`` si library_path no está configurado o el
        path queda fuera (anti traversal). El router traduce a HTTPException.
        """
        lib = self._library_path_loader()
        if not lib:
            raise ValueError("library_path no está configurado")
        try:
            lib_resolved = Path(lib).resolve(strict=False)
            target = Path(raw_path).resolve(strict=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Path inválido: {exc}") from exc

        try:
            target.relative_to(lib_resolved)
        except ValueError as exc:
            raise PermissionError(f"Path fuera de library_path: {raw_path}") from exc
        return target

    # --- escaneo ---------------------------------------------------------

    def scan_tree(self, root: Path) -> dict[str, Any]:
        """Escanea ``root`` y clasifica artefactos. Lanza ``FileNotFoundError``
        si la ruta no existe."""
        if not root.exists():
            raise FileNotFoundError(f"Path no existe: {root}")

        orphan_srt: list[dict[str, Any]] = []
        old_dubbed: list[dict[str, Any]] = []
        temp_files: list[dict[str, Any]] = []
        empty_dirs: list[dict[str, Any]] = []

        for dirpath, dirnames, filenames in self._repo.walk(root):
            dir_p = Path(dirpath)

            if not filenames and not dirnames:
                inf = info(dir_p)
                if inf:
                    empty_dirs.append(inf)

            video_stems = {
                Path(f).stem for f in filenames
                if Path(f).suffix.lower() in VIDEO_EXTS
            }

            for fname in filenames:
                fpath = dir_p / fname
                suffix = Path(fname).suffix.lower()

                if is_temp_file(fname):
                    inf = info(fpath)
                    if inf:
                        temp_files.append(inf)
                    continue

                if suffix == ".srt":
                    stem = Path(fname).stem
                    base_stem = stem.split(".")[0] if "." in stem else stem
                    if stem not in video_stems and base_stem not in video_stems:
                        inf = info(fpath)
                        if inf:
                            orphan_srt.append(inf)

                if suffix in VIDEO_EXTS and Path(fname).stem.endswith("_DOBLADO"):
                    orig_stem = Path(fname).stem[: -len("_DOBLADO")]
                    es_srt = dir_p / f"{orig_stem}.es.srt"
                    if not es_srt.exists():
                        es_srt = dir_p / f"{orig_stem}.ES.srt"
                    if es_srt.exists():
                        fstat = safe_stat(fpath)
                        sstat = safe_stat(es_srt)
                        if fstat and sstat and fstat[1] < sstat[1]:
                            inf = info(fpath)
                            if inf:
                                old_dubbed.append(inf)

        for lst in (orphan_srt, old_dubbed, temp_files, empty_dirs):
            lst.sort(key=lambda x: x["size"], reverse=True)

        total_items = len(orphan_srt) + len(old_dubbed) + len(temp_files) + len(empty_dirs)
        total_bytes = sum(
            i["size"] for i in (*orphan_srt, *old_dubbed, *temp_files, *empty_dirs)
        )

        return {
            "categories": {
                "orphan_srt": orphan_srt,
                "old_dubbed": old_dubbed,
                "temp_files": temp_files,
                "empty_dirs": empty_dirs,
            },
            "total_bytes": total_bytes,
            "total_items": total_items,
        }

    # --- background scan -------------------------------------------------

    def submit_scan(self, raw_path: str) -> str:
        """Encola un escaneo como background job. Devuelve ``job_id``."""
        target = self.resolve_under_library(raw_path)

        async def _coro(update_progress):
            update_progress(0.0, f"Scanning {target}...")

            def _work():
                return self.scan_tree(target)

            update_progress(10.0, "Walking filesystem...")
            result = await asyncio.to_thread(_work)
            update_progress(100.0, f"Found {result.get('total_items', 0)} items")
            return result

        job = self._jobs.submit("cleanup_scan", _coro, {"path": str(target)})
        return job.id

    # --- apply -----------------------------------------------------------

    def apply_deletions(
        self,
        paths: list[str],
        *,
        dry_run: bool,
    ) -> dict[str, Any]:
        """Borra los paths indicados revalidando cada uno.

        * Items que fallan NO abortan el resto; los errores van en ``errors``.
        * Vídeos ``.mkv``/``.mp4`` solo se borran si son ``_DOBLADO``.
        """
        deleted: list[str] = []
        errors: list[dict[str, str]] = []
        freed_bytes = 0

        for raw in paths:
            try:
                target = self.resolve_under_library(raw)
            except (ValueError, PermissionError) as exc:
                errors.append({"path": raw, "error": str(exc)})
                continue

            if not target.exists():
                errors.append({"path": raw, "error": "no existe"})
                continue

            suffix = target.suffix.lower()
            if target.is_file() and suffix in VIDEO_EXTS:
                if not target.stem.endswith("_DOBLADO"):
                    errors.append(
                        {"path": raw, "error": "vídeo no doblado: borrado denegado"}
                    )
                    continue

            try:
                if target.is_dir():
                    if not self._repo.is_dir_empty(target):
                        errors.append({"path": raw, "error": "directorio no vacío"})
                        continue
                    if not dry_run:
                        self._repo.delete_empty_dir(target)
                    deleted.append(str(target))
                else:
                    if not dry_run:
                        size = self._repo.delete_file(target)
                    else:
                        st = safe_stat(target)
                        size = st[0] if st else 0
                    deleted.append(str(target))
                    freed_bytes += int(size)
            except OSError as exc:
                errors.append({"path": raw, "error": str(exc)})

        return {
            "deleted": deleted,
            "errors": errors,
            "freed_bytes": int(freed_bytes),
            "dry_run": dry_run,
        }
