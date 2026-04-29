"""Persistent JSON store for jobs.

Single responsibility: serialize/deserialize jobs to/from disk.

Concurrencia: los endpoints de la API corren en el event loop y también
disparan background tasks; dos ``upsert`` concurrentes podían perder
actualizaciones (uno lee antes de que el otro escriba → overwrite).
Ahora todo acceso pasa por un ``threading.Lock`` y ``save`` escribe a
un archivo temporal y usa ``os.replace`` (atómico en el mismo volumen)
para evitar que un lector vea JSON truncado si la máquina se apaga.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class JobsStore:
    """Load/save a dict of jobs keyed by job_id as JSON on disk."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        # Lock per store instance. Distintos procesos siguen pudiendo
        # pisarse (esto es in-process), pero processor-api corre como
        # un único worker Uvicorn → basta con thread-level.
        self._lock = threading.Lock()

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            log.warning("jobs file %s not a dict, ignoring", self.path)
            return {}
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load jobs from %s: %s", self.path, exc)
            return {}

    def save(self, jobs: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            self._save_unlocked(jobs)

    def _save_unlocked(self, jobs: dict[str, dict[str, Any]]) -> None:
        self._ensure_dir()
        payload = json.dumps(jobs, indent=2, ensure_ascii=False, default=str)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)

    def upsert(self, job_id: str, job_data: dict[str, Any]) -> None:
        with self._lock:
            jobs = self._load_unlocked()
            jobs[job_id] = job_data
            self._save_unlocked(jobs)
