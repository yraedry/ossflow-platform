"""Módulo jobs: registry unificado de jobs en background.

Conviven dos sub-sistemas por compatibilidad histórica con la API HTTP
existente, **NO porque sean conceptualmente distintos**:

* **BackgroundJobsService** (BD SQLite, tabla ``background_jobs``):
  jobs largos asíncronos lanzados por ``cleanup``, ``duplicates`` y otros.
  Persisten progreso a BD, sobreviven al ciclo de vida del request
  (``threading.Thread`` con ``asyncio.run`` propio). Endpoint
  ``/api/background-jobs/*``.

* **LegacyJobsService** (BD SQLite, tabla ``legacy_jobs``): jobs estilo
  app.py viejo (chapter, subtitles, dubbing, elevenlabs). Mantienen el
  contrato ``/api/jobs/*`` con SSE en ``/api/jobs/{id}/events``.
  Conservan el campo ``video_path: str`` tipado en raíz del payload
  porque el frontend lo espera así.

Ambos servicios viven detrás de la misma anatomía Vertical Slice
(``routers/`` + ``services/`` + ``repositories/``) y comparten primitivas
en ``_internal/`` (scheduler, SSE hub, recovery de huérfanos).

Diseño completo: ``docs/superpowers/specs/2026-04-30-ossflow-api-jobs-module-design.md``.
"""

from .models import (
    BackgroundJob,
    JobStatus,
    LegacyJob,
)

__all__ = [
    "BackgroundJob",
    "JobStatus",
    "LegacyJob",
]
