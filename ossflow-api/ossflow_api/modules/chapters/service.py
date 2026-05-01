"""Servicio de chapters: lógica de negocio para renombrar capítulos.

Operaciones soportadas:

* ``rename_one``: renombra un capítulo individual preservando el prefijo
  ``SNNeMM`` y arrastra todos sus sidecars (subs/dubs).
* ``rename_by_oracle``: dado el oracle de un instructional, recorre la
  carpeta de Season y renombra cada capítulo emparejado por
  ``(volume, episode)``. Soporta también el formato crudo del splitter
  ``vol-ep.ext``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ossflow_api.shared.exceptions import (
    ApiError,
    ConflictError,
    NotFoundError,
    ValidationError,
)
from ossflow_api.shared.paths import to_container_path

from .repository import ChaptersRepository
from .schemas import (
    ILLEGAL_RE,
    MAX_TITLE_LEN,
    RAW_RE,
    SNNEMM_RE,
    VIDEO_EXTS,
    WS_RE,
)

log = logging.getLogger(__name__)


class _Forbidden(ApiError):
    """Path traversal u otro acceso fuera del library_path."""

    status_code = 403


class ChaptersService:
    """Encapsula validaciones, sanitizado y orquestación del repositorio."""

    def __init__(
        self,
        library_path: str | None,
        repository: ChaptersRepository | None = None,
    ) -> None:
        self._library_path = library_path
        self._repo = repository or ChaptersRepository()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_title(raw: str) -> str:
        """Recorta, sustituye chars ilegales por ``_``, colapsa espacios y limita.

        Devuelve cadena vacía si tras sanitizar queda vacío; el caller lo
        trata como fallo de validación.
        """
        if not isinstance(raw, str):
            return ""
        s = raw.strip()
        s = ILLEGAL_RE.sub("_", s)
        s = WS_RE.sub(" ", s).strip()
        if not s:
            return ""
        if len(s) > MAX_TITLE_LEN:
            s = s[:MAX_TITLE_LEN].rstrip()
        return s

    def _require_library_root(self) -> Path:
        if not self._library_path:
            raise ValidationError("library_path not configured", status_code=400)
        return Path(self._library_path)

    @staticmethod
    def _resolve_within_library(candidate: Path, library_root: Path) -> Path:
        """Devuelve el path absoluto resuelto o lanza 403 si escapa del root."""
        try:
            resolved = candidate.resolve(strict=False)
            root_resolved = library_root.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise _Forbidden(f"Path traversal: {exc}") from exc

        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise _Forbidden("Path traversal: target escapes library_path") from exc
        return resolved

    # ------------------------------------------------------------------
    # PATCH /api/chapters/rename
    # ------------------------------------------------------------------

    def rename_one(self, body: Any) -> dict[str, list[dict[str, str]]]:
        """Renombra un capítulo (y sus hermanos) preservando ``SNNeMM``.

        Espera ``{"old_path": str, "new_title": str}``. Devuelve la lista
        de archivos renombrados.
        """
        if not isinstance(body, dict):
            raise ValidationError("Body must be an object", status_code=422)

        old_path = body.get("old_path")
        new_title_raw = body.get("new_title")

        if not isinstance(old_path, str) or not old_path:
            raise ValidationError("old_path is required", status_code=422)
        if not isinstance(new_title_raw, str):
            raise ValidationError("new_title is required", status_code=422)

        sanitized = self._sanitize_title(new_title_raw)
        if not sanitized:
            raise ValidationError(
                "new_title is empty after sanitization",
                status_code=422,
            )

        library_root = self._require_library_root()

        old = Path(old_path)
        resolved_old = self._resolve_within_library(old, library_root)

        if not resolved_old.exists():
            raise NotFoundError(f"File not found: {old_path}")
        if not resolved_old.is_file():
            raise NotFoundError("old_path is not a file")

        # Parse SNNeMM del nombre de archivo (no del path completo).
        m = SNNEMM_RE.match(resolved_old.name)
        if not m:
            raise ValidationError(
                "Filename does not match `{prefix} - SNNeMM - {title}{ext}` pattern",
                status_code=422,
            )

        prefix = m.group("prefix").strip()
        season = m.group("season")
        ep = m.group("ep")
        ext = m.group("ext")

        new_filename = f"{prefix} - S{season}E{ep} - {sanitized}{ext}"
        new_path = resolved_old.with_name(new_filename)
        # Comprobar de nuevo que el destino sigue dentro del library (defensivo).
        self._resolve_within_library(new_path, library_root)

        renamed: list[dict[str, str]] = []

        # Renombrar el archivo principal primero (idempotente si no cambia).
        if new_path != resolved_old:
            if new_path.exists():
                raise ConflictError(f"Target already exists: {new_path.name}")
            self._repo.rename(resolved_old, new_path)
        renamed.append({"from": str(resolved_old), "to": str(new_path)})

        # Renombrar hermanos (basado en old_stem → new_stem).
        old_stem = resolved_old.stem  # e.g. "Author - S01E01 - Old Title"
        new_stem = new_path.stem
        renamed.extend(
            self._repo.rename_siblings(resolved_old, old_stem, new_stem)
        )

        return {"renamed": renamed}

    # ------------------------------------------------------------------
    # POST /api/chapters/rename-by-oracle
    # ------------------------------------------------------------------

    def rename_by_oracle(self, body: Any) -> dict[str, list]:
        """Renombra todos los capítulos de una Season usando el oracle.

        Empareja el ``SNNeMM`` (o el formato crudo ``vol-ep``) con el
        ``(volume, episode)`` del oracle y reemplaza la porción del título.
        Los archivos sin match en el oracle se devuelven en ``skipped``.
        """
        if not isinstance(body, dict):
            raise ValidationError("Body must be an object", status_code=422)

        season_path_str = body.get("season_path")
        oracle = body.get("oracle")

        if not isinstance(season_path_str, str) or not season_path_str:
            raise ValidationError("season_path is required", status_code=422)
        if not isinstance(oracle, dict):
            raise ValidationError("oracle is required", status_code=422)

        library_root = self._require_library_root()
        library_root_str = str(self._library_path or "")

        container_path_str = to_container_path(season_path_str, library_root_str)
        season_dir = Path(container_path_str)
        resolved_season = self._resolve_within_library(season_dir, library_root)
        if not resolved_season.exists() or not resolved_season.is_dir():
            raise NotFoundError(
                f"Season directory not found: {container_path_str}"
            )

        # Construir mapping (volume_num, episode_num) → title desde el oracle.
        # Estructura oracle:
        #   {"volumes": [{"number": N, "chapters": [{"number": M, "title": "..."}]}]}
        oracle_map: dict[tuple[int, int], str] = {}
        for vol in oracle.get("volumes", []):
            vol_num = int(vol.get("number", 0))
            for idx, ch in enumerate(vol.get("chapters", []), start=1):
                ep_num = int(ch.get("number", idx))
                title = ch.get("title", "").strip()
                if title:
                    oracle_map[(vol_num, ep_num)] = title

        if not oracle_map:
            raise ValidationError(
                "Oracle contains no chapter titles",
                status_code=422,
            )

        renamed: list[dict[str, str]] = []
        skipped: list[str] = []

        instructional_name = body.get("instructional_name", "").strip()

        for f in self._repo.iter_dir(resolved_season):
            if not f.is_file() or f.suffix.lower() not in VIDEO_EXTS:
                continue

            m = SNNEMM_RE.match(f.name)
            if m:
                season_num = int(m.group("season"))
                ep_num = int(m.group("ep"))
                prefix = m.group("prefix").strip()
                season_str = m.group("season")
                ep_str = m.group("ep")
                ext = m.group("ext")
            else:
                m2 = RAW_RE.match(f.name)
                if not m2:
                    skipped.append(f.name)
                    continue
                season_num = int(m2.group("vol"))
                ep_num = int(m2.group("ep"))
                prefix = instructional_name
                season_str = f"{season_num:02d}"
                ep_str = f"{ep_num:02d}"
                ext = m2.group("ext")

            oracle_title = oracle_map.get((season_num, ep_num))
            if oracle_title is None:
                skipped.append(f.name)
                continue

            sanitized = self._sanitize_title(oracle_title)
            if not sanitized:
                skipped.append(f.name)
                continue

            sep = " - " if prefix else ""
            new_filename = (
                f"{prefix}{sep}S{season_str}E{ep_str} - {sanitized}{ext}"
            )
            new_path = f.with_name(new_filename)

            if new_path == f:
                continue
            if new_path.exists():
                log.warning(
                    "rename-by-oracle: target exists, skipping: %s", new_path.name
                )
                skipped.append(f.name)
                continue

            self._resolve_within_library(new_path, library_root)
            old_stem = f.stem
            new_stem = new_path.stem

            self._repo.rename(f, new_path)
            renamed.append({"from": str(f), "to": str(new_path)})

            renamed.extend(
                self._repo.rename_siblings(f, old_stem, new_stem)
            )

        return {"renamed": renamed, "skipped": skipped}
