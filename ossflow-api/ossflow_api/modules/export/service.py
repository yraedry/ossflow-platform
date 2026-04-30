"""Servicio del módulo export.

Envuelve dos backends de exportación:

* ``OssFlowClient`` (paquete ``ossflow_client``) — exporta un
  instructional completo a un servidor OssFlow remoto.
* ``PlexExporter`` (paquete ``chapter_tools.plex_exporter``) —
  fragmenta un vídeo por capítulos en el formato compatible con
  Plex.

Las dos clases se inyectan vía factories para que los tests puedan
sustituirlas sin tocar I/O ni red.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from ossflow_api.shared.exceptions import ApiError, NotFoundError, ValidationError

log = logging.getLogger(__name__)


# Factories: callable que devuelven instancias listas para usar.
# OssFlow client recibe ``base_url`` opcional; Plex no recibe nada.
OssFlowClientFactory = Callable[..., Any]
PlexExporterFactory = Callable[[], Any]


class ExportService:
    """Operaciones de exportación a OssFlow y a Plex."""

    def __init__(
        self,
        *,
        ossflow_factory: OssFlowClientFactory,
        plex_factory: PlexExporterFactory,
    ) -> None:
        self._ossflow_factory = ossflow_factory
        self._plex_factory = plex_factory

    # ------------------------------------------------------------------
    # OssFlow
    # ------------------------------------------------------------------

    def export_to_ossflow(
        self,
        *,
        path: str,
        instructor: str,
        base_url: str,
    ) -> dict[str, Any]:
        """Exporta un instructional al backend OssFlow."""
        if not path or not instructor:
            raise ValidationError("Missing 'path' or 'instructor'")
        if not Path(path).exists():
            raise NotFoundError("Path does not exist")

        try:
            client = self._ossflow_factory(base_url=base_url)
            summary = client.export_full_instructional(Path(path), instructor)
            return {"ok": True, "summary": summary}
        except Exception as exc:  # noqa: BLE001
            log.error("export_to_ossflow failed: %s", exc)
            raise ApiError(str(exc), status_code=500) from exc

    def ossflow_status(self) -> dict[str, Any]:
        """Comprueba si el backend OssFlow es alcanzable.

        Mantiene el shape del legacy: nunca lanza, devuelve
        ``{"reachable": bool, "error"?: str}``.
        """
        try:
            client = self._ossflow_factory()
            reachable = client.health_check()
            return {"reachable": reachable}
        except Exception as exc:  # noqa: BLE001
            log.error("ossflow_status failed: %s", exc)
            return {"reachable": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Plex
    # ------------------------------------------------------------------

    def export_to_plex(
        self,
        *,
        name: str,
        chapters: list[Any],
        source_dir: str,
        output_dir: str,
    ) -> dict[str, Any]:
        """Exporta un instructional a formato compatible con Plex."""
        if not name or not chapters or not source_dir or not output_dir:
            raise ValidationError(
                "Missing required fields: name, chapters, source_dir, output_dir"
            )
        if not Path(source_dir).exists():
            raise NotFoundError("source_dir does not exist")

        try:
            exporter = self._plex_factory()
            exporter.export(
                name, chapters, Path(source_dir), Path(output_dir)
            )
            return {
                "ok": True,
                "message": f"Exported '{name}' to {output_dir}",
            }
        except FileNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            log.error("export_plex failed: %s", exc)
            raise ApiError(str(exc), status_code=500) from exc
