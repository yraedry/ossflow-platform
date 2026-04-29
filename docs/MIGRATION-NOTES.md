# Notas del Split de Repos (2026-04-29)

## Cambios principales

- Monorepo `ossflow-scrapper-bck` (originalmente `bjj-processor-v2`) dividido en 4 repos.
- Identificadores `bjj_*` renombrados a `ossflow_*`.
- Imagen Docker base `bjj-base` → `ossflow-base`.
- Paquete Python `bjj_service_kit` → `ossflow_service_kit`, distribuido vía Git tag.
- Network compose `bjj_net` → `ossflow_net`.
- Variable env `BJJ_DB_PATH` → `OSSFLOW_DB_PATH` (con fallback a `BJJ_DB_PATH` durante una transición).
- Servicios renombrados:
  - `processor-api` → `ossflow-api`
  - `chapter-splitter` (signal) → `ossflow-splitter`
  - `chapter-splitter` (oracle / lean) → repo independiente `ossflow-scrapper`
  - `subtitle-generator` → `ossflow-subtitle`
  - `dubbing-generator` → `ossflow-dubbing`
  - `telegram-fetcher` → `ossflow-telegram`
  - `processor-frontend` → repo independiente `ossflow-studio` / `ossflow-frontend`
- Eliminado el modulo `burn_subs` (funcionalidad absorbida por `dubbing`).
- Eliminado el servicio `ollama` interno (vive en LXC externo `10.10.100.13`).
- `OLLAMA_BASE_URL` ahora obligatoria en `.env`.

## Pendiente (proximos planes)

- Refactor interno de `ossflow-api` con Vertical Slice (ver spec seccion 3.3).
- Refactor interno de los demas servicios segun su patron asignado (ver spec seccion 3.2).
- Eliminar el fallback `BJJ_DB_PATH` cuando ya no haya entornos legacy.
- Renombrar el archivo fisico de la BD `bjj.db` → `ossflow.db`.
- Borrar `tests/test_compose.py` de `ossflow-core` (vive en sitio equivocado, ensucia la suite).
- Migrar el kit `ossflow-core/ossflow_service_kit/db/engine.py` para que tambien use `OSSFLOW_DB_PATH` con fallback (hoy solo lee `BJJ_DB_PATH`).
