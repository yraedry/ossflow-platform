"""Dependencias FastAPI del módulo export."""

from __future__ import annotations

from .service import ExportService


def _default_ossflow_factory(*, base_url: str | None = None):
    """Crea un ``OssFlowClient`` por request.

    Import diferido — ``ossflow_client`` vive como paquete sibling y se
    instala via ``requirements.txt``.
    """
    from ossflow_client.client import OssFlowClient, OssFlowConfig

    if base_url is not None:
        return OssFlowClient(OssFlowConfig(base_url=base_url))
    return OssFlowClient()


def _default_plex_factory():
    """Crea un ``PlexExporter`` por request."""
    from chapter_tools.plex_exporter import PlexExporter

    return PlexExporter()


def get_export_service() -> ExportService:
    return ExportService(
        ossflow_factory=_default_ossflow_factory,
        plex_factory=_default_plex_factory,
    )
