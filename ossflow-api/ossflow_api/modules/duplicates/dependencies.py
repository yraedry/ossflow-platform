"""Dependencias FastAPI del módulo duplicates."""

from __future__ import annotations

from fastapi import Depends

from ossflow_api.modules.jobs.dependencies import get_background_jobs_service
from ossflow_api.modules.jobs.services.background import BackgroundJobsService

from .service import DuplicatesService


def get_duplicates_service(
    jobs: BackgroundJobsService = Depends(get_background_jobs_service),
) -> DuplicatesService:
    # Imports diferidos: settings + get_video_info viven aún en api.* hasta T23/F5.
    from api.app import get_video_info
    from api.settings import get_library_path

    return DuplicatesService(
        jobs=jobs,
        library_path_loader=get_library_path,
        video_info_loader=get_video_info,
    )
