"""Dependencias FastAPI del módulo metadata."""

from __future__ import annotations

from .service import MetadataService

# Import diferido para evitar import circular durante la migración.
def get_metadata_service() -> MetadataService:
    # api.settings.get_library_path es la fuente única de verdad mientras
    # settings no se haya migrado. Cuando se migre, este import cambiará.
    from api.settings import get_library_path
    return MetadataService(get_library_path())
