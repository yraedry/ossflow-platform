#!/bin/bash
set -e

# Arrancar ollama en background
/bin/ollama serve &
OLLAMA_PID=$!

# Esperar a que el servidor responda
until curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; do
  sleep 2
done

# Pull idempotente del modelo — grep en subshell para no activar set -e
MODEL="qwen2.5:7b-instruct-q4_K_M"
if ! (curl -fsS http://localhost:11434/api/tags | grep -q "$MODEL"); then
  echo "Pulling $MODEL (primer arranque, ~4.5 GB)..."
  ollama pull "$MODEL"
fi

wait $OLLAMA_PID
