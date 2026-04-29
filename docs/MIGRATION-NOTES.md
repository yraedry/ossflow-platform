# Notas del Split de Repos (2026-04-29)

## Cambios principales

- Monorepo `ossflow-scrapper-bck` (originalmente `bjj-processor-v2`) dividido en 4 repos.
- Identificadores `bjj_*` renombrados a `ossflow_*`.
- Imagen Docker base `bjj-base` → `ossflow-base`.
- Paquete Python `bjj_service_kit` → `ossflow_service_kit`, distribuido vía Git tag (`ossflow-core@v0.1.0`).
- Network compose `bjj_net` → `ossflow_net`.
- Variable env `BJJ_DB_PATH` → `OSSFLOW_DB_PATH` (con fallback durante transición).
- Servicios renombrados:
  - `processor-api` → `ossflow-api`
  - `chapter-splitter` (signal) → `ossflow-splitter`
  - `chapter-splitter` (oracle / lean) → repo independiente `ossflow-scrapper`
  - `subtitle-generator` → `ossflow-subtitle`
  - `dubbing-generator` → `ossflow-dubbing`
  - `telegram-fetcher` → `ossflow-telegram`
  - `processor-frontend` → repo independiente `ossflow-studio`
- Eliminado el servicio `ollama` interno (vive en LXC externo `10.10.100.13`).
- `OLLAMA_BASE_URL` ahora obligatoria en `.env`.

## Pendiente (próximos planes)

- Refactor interno con Vertical Slice (ver spec sección 3.3).
- Eliminar fallback `BJJ_DB_PATH` cuando ya no haya entornos legacy.
- Renombrar archivo físico `bjj.db` → `ossflow.db`.
- Borrar `tests/test_compose.py` de `ossflow-core` (vive en sitio equivocado).
- Migrar `ossflow-core/ossflow_service_kit/db/engine.py` para que use `OSSFLOW_DB_PATH` con fallback.
- Build pesado de subtitle/dubbing en el LXC (no testeable localmente sin GPU + modelos descargados).
