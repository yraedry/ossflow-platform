# Lean image for the telegram-fetcher service.
# Telethon client + FastAPI + aiosqlite. No ffmpeg (the splitter handles media).
# Expected size ~250MB.
FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Shared service kit first (small, stable) — provides app factory, /health, /logs.
COPY bjj_service_kit /opt/bjj_service_kit
RUN pip install --no-cache-dir /opt/bjj_service_kit

# Python deps for telegram-fetcher.
COPY telegram-fetcher/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the telegram_fetcher package + entrypoints (app.py created by T3).
COPY telegram-fetcher /app

# Runtime state: Telethon session + download cache.
RUN mkdir -p /data/session /data/cache \
    && chmod -R 0777 /data

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=15s \
    CMD curl -fsS http://localhost:8004/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
