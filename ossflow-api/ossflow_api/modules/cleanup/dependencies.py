"""Dependencias FastAPI del módulo cleanup."""

from __future__ import annotations

from fastapi import Depends

from ossflow_api.modules.jobs.dependencies import get_background_jobs_service
from ossflow_api.modules.jobs.services.background import BackgroundJobsService

from .repository import CleanupRepository
from .service import CleanupService


def get_cleanup_service(
    jobs: BackgroundJobsService = Depends(get_background_jobs_service),
) -> CleanupService:
    # Import diferido a settings (mismo patrón que cleanup/duplicates en
    # el resto de módulos migrados).
    from api.settings import get_library_path

    return CleanupService(
        repo=CleanupRepository(),
        jobs=jobs,
        library_path_loader=get_library_path,
    )
