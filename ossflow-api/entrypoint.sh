#!/bin/sh
# Entrypoint del contenedor ossflow-api.
#
# 1. Ejecuta el script idempotente de migración JSON → SQL.
#    En contenedores donde no haya .json viejos, es no-op (~100 ms).
#    En contenedores con backups restaurados, importa los datos legacy
#    a la BD unificada y renombra los archivos a .bak.
# 2. Lanza uvicorn.
#
# Cualquier fallo del script de migración NO bloquea el arranque
# (logueamos el error y seguimos), porque la app puede operar con BD
# vacía y los .json siguen en disco para reintentar al siguiente
# arranque.

set -e

CONFIG_DIR="${CONFIG_DIR:-/data/config}"

echo "[entrypoint] Running migration script (config-dir=$CONFIG_DIR)..."
python -m scripts.migrate_json_to_db --config-dir "$CONFIG_DIR" || {
    echo "[entrypoint] WARNING: migration script failed; continuing anyway"
}

echo "[entrypoint] Launching uvicorn..."
exec uvicorn api.app:app --host 0.0.0.0 --port 8000 "$@"
