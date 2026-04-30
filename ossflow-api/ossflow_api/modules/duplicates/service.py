"""Servicio de detección de duplicados.

Heurística rápida: dos vídeos son candidatos duplicados cuando comparten
``(size_bytes, duration_seconds_rounded_to_1s)``. Modo ``deep`` añade un
md5 parcial de los primeros 10 MB.

Mantiene la lógica idéntica al antiguo ``api/duplicates.py``. Recibe
``BackgroundJobsService`` por DI (cierre acoplamiento #3).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from ossflow_api.modules.jobs.services.background import BackgroundJobsService

log = logging.getLogger(__name__)

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")
DEEP_SAMPLE_BYTES = 10 * 1024 * 1024  # 10 MB


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:  # noqa: BLE001
        return False


def _partial_md5(path: Path, nbytes: int = DEEP_SAMPLE_BYTES) -> Optional[str]:
    try:
        h = hashlib.md5()
        with path.open("rb") as fh:
            chunk = fh.read(nbytes)
            h.update(chunk)
        return h.hexdigest()
    except Exception as exc:  # noqa: BLE001
        log.warning("md5 failed for %s: %s", path, exc)
        return None


VideoInfoLoader = Callable[[str], dict]


class DuplicatesService:
    """Detector de duplicados con DI de jobs y video info loader."""

    def __init__(
        self,
        jobs: BackgroundJobsService,
        library_path_loader: Callable[[], Optional[str]],
        video_info_loader: VideoInfoLoader,
    ) -> None:
        self._jobs = jobs
        self._library_path_loader = library_path_loader
        self._video_info_loader = video_info_loader

    # --- validación -----------------------------------------------------

    def validate_path(self, raw_path: str) -> Path:
        """Valida que ``raw_path`` esté bajo ``library_path`` y exista.

        Lanza ``ValueError`` (lib no configurada / path inexistente) o
        ``PermissionError`` (path fuera de la librería). El router
        traduce a HTTPException.
        """
        lib = self._library_path_loader()
        if not lib:
            raise ValueError("library_path no configurado")
        root = Path(raw_path)
        if not _is_under(root, Path(lib)):
            raise PermissionError("Path fuera de la librería")
        if not root.exists() or not root.is_dir():
            raise ValueError("Path no existe")
        return root

    # --- scan -----------------------------------------------------------

    def scan(
        self,
        root: Path,
        *,
        deep: bool = False,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> dict[str, Any]:
        candidates: list[Path] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(VIDEO_EXTS):
                    candidates.append(Path(dirpath) / fname)

        total_candidates = len(candidates)
        if progress_cb:
            progress_cb(5.0, f"Found {total_candidates} video candidates")

        signatures: dict[tuple[int, int], list[dict[str, Any]]] = {}
        total_videos = 0

        for idx, fpath in enumerate(candidates):
            try:
                size = fpath.stat().st_size
            except OSError as exc:
                log.warning("stat failed for %s: %s", fpath, exc)
                continue
            try:
                info = self._video_info_loader(str(fpath))
            except Exception as exc:  # noqa: BLE001
                log.warning("ffprobe failed for %s: %s", fpath, exc)
                continue
            duration = float(info.get("duration", 0) or 0)
            if duration <= 0:
                continue
            total_videos += 1
            key = (size, int(round(duration)))
            signatures.setdefault(key, []).append({
                "path": str(fpath),
                "size": size,
                "duration_sec": int(round(duration)),
            })
            if progress_cb and total_candidates:
                pct = 5.0 + 85.0 * ((idx + 1) / total_candidates)
                progress_cb(pct, f"Probed {idx + 1}/{total_candidates}")

        groups = [g for g in signatures.values() if len(g) >= 2]

        if deep and groups:
            if progress_cb:
                progress_cb(92.0, "Computing partial md5 for candidate groups")
            confirmed: list[list[dict[str, Any]]] = []
            for group in groups:
                by_hash: dict[str, list[dict[str, Any]]] = {}
                for entry in group:
                    digest = _partial_md5(Path(entry["path"]))
                    if digest is None:
                        continue
                    by_hash.setdefault(digest, []).append(entry)
                for entries in by_hash.values():
                    if len(entries) >= 2:
                        confirmed.append(entries)
            groups = confirmed

        wasted_bytes = sum(e["size"] for group in groups for e in group[1:])

        if progress_cb:
            progress_cb(100.0, f"Done: {len(groups)} duplicate groups")

        return {
            "groups": groups,
            "stats": {
                "total_videos": total_videos,
                "groups_found": len(groups),
                "wasted_bytes": wasted_bytes,
            },
        }

    # --- background scan ------------------------------------------------

    def submit_scan(self, raw_path: str, *, deep: bool = False) -> str:
        root = self.validate_path(raw_path)

        async def _coro(update_progress):
            update_progress(0.0, f"Scanning {root} for duplicates...")

            def _work():
                return self.scan(
                    root,
                    deep=deep,
                    progress_cb=lambda p, m: update_progress(p, m),
                )

            return await asyncio.to_thread(_work)

        job = self._jobs.submit(
            "duplicates_scan", _coro, {"path": str(root), "deep": bool(deep)}
        )
        return job.id
