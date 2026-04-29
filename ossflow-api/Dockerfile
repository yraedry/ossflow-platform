# syntax=docker/dockerfile:1.7
# Build context: ../python (repo root) so bjj_service_kit is visible.
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg mkvtoolnix cifs-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Shared service kit (DB layer, app factory, logs ring buffer).
COPY bjj_service_kit /opt/bjj_service_kit
RUN pip install --no-cache-dir /opt/bjj_service_kit

COPY processor-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt requests

COPY processor-api /app
VOLUME /data
EXPOSE 8000
ENV PYTHONPATH=/app
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
