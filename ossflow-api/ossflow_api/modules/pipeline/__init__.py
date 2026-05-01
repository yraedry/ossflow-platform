"""Módulo pipeline: orquestador del workflow de un instructional.

Migración en curso desde ``api/pipeline.py`` (1723 LOC monolíticas) a
una descomposición vertical-slice estándar. Plan T_LATE_2:

* ``schemas.py`` — ``StepStatus``, ``StepInfo``, ``PipelineInfo``,
  ``BatchInfo``, ``VALID_STEPS``, ``STEP_ORDER``, serializadores.
* ``history.py`` — persistencia debounced del JSON de historia.
* ``store.py`` — state container (in-memory) con SSE fan-out.
* ``skip_detector.py`` — heurísticas para saltar pasos ya hechos.
* ``diff.py`` — snapshot de directorios + diff (ficheros nuevos).
* ``backend_dispatch.py`` — mapping step → microservicio HTTP.
* ``runner.py`` — service: ejecuta steps, pipelines y batches.
* ``eta.py`` — estimación de duración.
* ``router.py`` — endpoints HTTP + SSE consumer.
* ``dependencies.py`` — DI factories (singleton store + runner).
* ``service.py`` — fachada delgada que reexporta runner.

Durante la migración ``api/pipeline.py`` se mantiene como shim de
re-export. Los tests parchean símbolos privados por nombre desde el
shim, así que cada sub-tarea T_LATE_2.* debe preservar la
visibilidad de esos símbolos hasta T_LATE_2.7 (cleanup).
"""
