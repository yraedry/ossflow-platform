"""Compat shim. Lógica movida a ``ossflow_api.modules.jobs``.

Este shim mantiene la API pública legacy (``registry``, ``BackgroundJob``,
``QUEUED/RUNNING/COMPLETED/FAILED``, ``router``, ``JobRegistry`` clase)
para que ``cleanup``, ``duplicates`` y ``burn_subs`` sigan funcionando
sin cambios hasta T20-T22, donde se migran al nuevo patrón Vertical
Slice e importan ``BackgroundJobsService`` directamente.

Cuando esos tres módulos hayan migrado, este shim se elimina.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ossflow_api.modules.jobs.dependencies import (  # noqa: F401
    get_background_jobs_service,
)
from ossflow_api.modules.jobs.models import BackgroundJob, JobStatus  # noqa: F401
from ossflow_api.modules.jobs.repositories.background import BackgroundJobsRepository
from ossflow_api.modules.jobs.routers.background import router  # noqa: F401
from ossflow_api.modules.jobs.services.background import BackgroundJobsService

# Constantes de status (re-exportadas con los nombres legacy).
QUEUED = JobStatus.QUEUED.value
RUNNING = JobStatus.RUNNING.value
COMPLETED = JobStatus.COMPLETED.value
FAILED = JobStatus.FAILED.value


# Alias retrocompat: ``JobRegistry`` legacy → ``BackgroundJobsService``.
# La API pública (``submit``, ``get``, ``list_all``) coincide. ``init`` se
# llama explícitamente para mantener el comportamiento del constructor
# legacy que cargaba al instanciar.
class JobRegistry(BackgroundJobsService):
    """Compat shim: misma API que ``BackgroundJobsService`` con auto-init.

    El ``JobRegistry`` legacy hacía ``self._load()`` en ``__init__``. Lo
    replicamos aquí para no romper tests que esperan ese comportamiento.
    """

    def __init__(self, history_file: Optional[Path] = None) -> None:
        from ossflow_api.modules.jobs._internal.scheduler import JobsScheduler

        repo = BackgroundJobsRepository(history_file=history_file)
        super().__init__(repo=repo, scheduler=JobsScheduler())
        self.init()


# Singleton legacy: el ``registry`` que importan cleanup/duplicates/burn_subs.
# Apunta al servicio singleton scope-app del módulo nuevo.
registry = get_background_jobs_service()
registry.init()
