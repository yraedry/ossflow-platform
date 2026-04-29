# syntax=docker/dockerfile:1.7
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg mkvtoolnix cifs-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Shared service kit (DB layer, app factory, logs ring buffer).
RUN pip install --no-cache-dir "ossflow-service-kit @ git+https://github.com/yraedry/ossflow-core@v0.1.0#subdirectory=ossflow_service_kit"

COPY processor-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt requests

COPY processor-api /app
VOLUME /data
EXPOSE 8000
ENV PYTHONPATH=/app
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
