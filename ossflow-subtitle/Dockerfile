# syntax=docker/dockerfile:1.7
FROM ossflow-base:latest

# 1. Solo requirements.txt — esta layer solo se invalida si cambia requirements.txt
COPY ossflow-subtitle/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,id=ossflow-pip,sharing=locked \
    pip install -r /app/requirements.txt \
 && pip uninstall -y pytest pytest-asyncio || true \
 && find /opt/venv -depth -type d -name "__pycache__" -exec rm -rf {} + \
 && find /opt/venv -type f -name "*.pyc" -delete \
 && find /opt/venv/lib/python3.11/site-packages -maxdepth 3 -type d \
      \( -name "tests" -o -name "test" -o -name "docs" -o -name "examples" \) \
      -exec rm -rf {} + 2>/dev/null || true

# 2. ossflow-service-kit — instalado desde Git tras los deps principales
RUN pip install --no-cache-dir "ossflow-service-kit @ git+https://github.com/yraedry/ossflow-core@v0.1.0#subdirectory=ossflow_service_kit"

# 3. Código del servicio — cambia frecuentemente, no invalida nada de arriba
COPY ossflow-subtitle /app/ossflow-subtitle
WORKDIR /app/ossflow-subtitle

EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8002/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
