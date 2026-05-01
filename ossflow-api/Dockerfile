# syntax=docker/dockerfile:1.7
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg mkvtoolnix cifs-utils git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Shared service kit (DB layer, app factory, logs ring buffer).
RUN pip install --no-cache-dir "ossflow-service-kit @ git+https://github.com/yraedry/ossflow-core@v0.1.0#subdirectory=ossflow_service_kit"

COPY ossflow-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt requests

COPY ossflow-api /app
RUN chmod +x /app/entrypoint.sh
VOLUME /data
EXPOSE 8000
ENV PYTHONPATH=/app
# El entrypoint ejecuta el script idempotente de migración JSON → SQL
# antes de lanzar uvicorn. Migra settings.json + background_jobs.json +
# jobs.json a la BD unificada en cualquier despliegue (incluyendo LXC
# nuevo o backup restaurado). Sin .json viejos, es no-op de ~100 ms.
ENTRYPOINT ["/app/entrypoint.sh"]
