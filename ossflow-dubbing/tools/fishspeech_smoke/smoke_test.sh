#!/usr/bin/env bash
# Smoke test del runtime oficial fish-speech sobre la GPU del LXC.
#
# Objetivo: confirmar antes de migrar ossflow-dubbing que el runtime
# Python+PyTorch oficial de Fish Audio:
#   1. Arranca dentro de un container con la 2060 6GB.
#   2. Carga S2-Pro sin OOM (safetensors bf16 + cualquier optimización
#      que su CLI permita por defecto).
#   3. Sintetiza una frase ES con voice clone (ref WAV + transcript).
#   4. Latencia y util GPU son razonables (objetivo: <15s/frase, util
#      sostenida >50%).
#
# Tras correr esto, decidimos si la migración merece la pena o si la
# 2060 se queda corta y hay que ir a alternativa hardware.
#
# Uso (en el LXC):
#   bash smoke_test.sh
#
# Variables override:
#   FISH_REF=v1.5.0      # tag o SHA del repo fishaudio/fish-speech
#   VOICE_WAV=...        # path host al WAV de referencia (default: el
#                          que usas en producción)
#   VOICE_TXT=...        # transcripción del WAV (¡debe coincidir!)
#   TEST_TEXT=...        # frase a sintetizar
#   MODELS_DIR=...       # dónde se cachean los safetensors descargados

set -euo pipefail

FISH_REF="${FISH_REF:-main}"
VOICES_DIR="${VOICES_DIR:-/opt/ossflow/ossflow-platform/ossflow-dubbing/voices}"
VOICE_WAV="${VOICE_WAV:-${VOICES_DIR}/craig_16s.wav}"
# Si no se pasa VOICE_TXT, intentamos leer el sidecar .txt con el mismo
# stem que el WAV (la convención del propio dubbing-generator —
# `_transcript_path_for` en api/router.py).
if [ -z "${VOICE_TXT:-}" ]; then
    SIDECAR="${VOICE_WAV%.*}.txt"
    if [ -f "$SIDECAR" ]; then
        VOICE_TXT="$(cat "$SIDECAR")"
    fi
fi
VOICE_TXT="${VOICE_TXT:-}"
TEST_TEXT="${TEST_TEXT:-Cuando encuentres un problema al intentarlo en el entrenamiento, vuelve directamente a la posición anterior.}"
MODELS_DIR="${MODELS_DIR:-/opt/ossflow/ossflow-platform/models/fish-speech-cache}"
OUT_WAV="${OUT_WAV:-/tmp/fishspeech_smoke_output.wav}"

echo "=== fish-speech smoke test ==="
echo "FISH_REF:    $FISH_REF"
echo "VOICE_WAV:   $VOICE_WAV"
echo "TEST_TEXT:   $TEST_TEXT"
echo "MODELS_DIR:  $MODELS_DIR"
echo "OUT_WAV:     $OUT_WAV"
echo

# Sanity checks
if [ ! -f "$VOICE_WAV" ]; then
    echo "ERROR: VOICE_WAV not found: $VOICE_WAV"
    echo "Set VOICE_WAV=... to your reference WAV path."
    exit 1
fi
if [ -z "$VOICE_TXT" ]; then
    echo "ERROR: VOICE_TXT empty. Set VOICE_TXT='exact transcript of VOICE_WAV'."
    echo "Voice cloning collapses if transcript and audio drift."
    exit 1
fi

mkdir -p "$MODELS_DIR"

# Disk check (modelo S2-Pro pesa ~10GB; build PyTorch tira ~5GB temp)
free_gb=$(df -BG --output=avail "$MODELS_DIR" | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt 15 ]; then
    echo "WARNING: only ${free_gb}GB free in $MODELS_DIR (need >=15GB)."
    echo "Press Ctrl+C to abort, or wait 5s to continue at your own risk..."
    sleep 5
fi

# Lanza el container. Mantengo nvidia-runtime + capabilities graphics
# (no hace falta para fish-speech pero el LXC ya lo tiene configurado y
# no hace daño).
#
# Estrategia de modelo:
#   - HF_HOME apunta al MODELS_DIR del host, así el download persiste
#     entre runs y no re-descargamos en cada smoke test.
#   - El runtime de fish-speech detecta el modelo automáticamente vía
#     `huggingface_hub` cuando le pasas el repo `fishaudio/s2-pro`.
#
# Comando dentro del container:
#   1. Clona fish-speech.
#   2. Instala torch + deps (la imagen base ya trae CUDA runtime).
#   3. Descarga modelo S2-Pro desde HF a HF_HOME.
#   4. Lanza CLI inferencia con voice clone.
#   5. Mide tiempo wall-clock y guarda WAV resultado.

docker run --rm \
    --runtime=nvidia \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$MODELS_DIR:/root/.cache/huggingface" \
    -v "$VOICE_WAV:/work/ref.wav:ro" \
    -v "$(dirname "$OUT_WAV"):/out" \
    -e FISH_REF="$FISH_REF" \
    -e VOICE_TXT="$VOICE_TXT" \
    -e TEST_TEXT="$TEST_TEXT" \
    -e OUT_NAME="$(basename "$OUT_WAV")" \
    nvidia/cuda:12.4.1-runtime-ubuntu22.04 bash -c '
set -euo pipefail

echo "--- Stage 1: deps ---"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# python3.10 viene en la imagen base; incluimos -dev por pyaudio (que
# necesita Python.h durante el build de su extensión C). portaudio19
# y libsndfile1 son las nativas.
apt-get install -y --no-install-recommends -qq \
    python3 python3-dev python3-pip python3-venv git ffmpeg \
    build-essential portaudio19-dev libsndfile1 \
    curl bc \
    > /dev/null 2>&1
ln -sf /usr/bin/python3 /usr/local/bin/python

echo "--- Stage 2: clone fish-speech @ $FISH_REF ---"
git clone --depth 50 https://github.com/fishaudio/fish-speech.git /work/fish-speech
cd /work/fish-speech
git checkout "$FISH_REF" 2>/dev/null || echo "(staying on default branch)"

echo "--- Stage 3: pip install ---"
pip install --no-cache-dir --upgrade pip > /dev/null
# torch primero con index CUDA 12.4
pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124 > /tmp/pip.log 2>&1 \
  || { echo "FAIL: torch install"; tail -30 /tmp/pip.log; exit 1; }
pip install --no-cache-dir -e . > /tmp/pip.log 2>&1 \
  || { echo "FAIL: fish-speech install"; tail -50 /tmp/pip.log; exit 1; }

echo "--- Stage 4: download S2-Pro from HF ---"
python -c "
from huggingface_hub import snapshot_download
import time
t0 = time.time()
path = snapshot_download(repo_id=\"fishaudio/s2-pro\", local_dir_use_symlinks=False)
print(f\"Downloaded to {path} in {time.time()-t0:.0f}s\")
"

echo "--- Stage 5: GPU sanity ---"
python <<PYGPU
import torch
ok = torch.cuda.is_available()
print(f"CUDA available: {ok}")
if ok:
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Device: {name}")
    print(f"VRAM total: {vram:.1f} GB")
else:
    print("Device: none")
PYGPU

echo "--- Stage 6: launch HTTP server + POST test phrase ---"
# tools.api_server es un servidor HTTP (Kui+uvicorn) idéntico en
# patrón a s2.cpp --server. Lo arrancamos en background, esperamos a
# que esté ready en su puerto, y mandamos un POST con la frase de
# test + ref WAV + transcript. Mide latencia wall-clock del POST y
# VRAM tras el primer hit.

API_PORT="${API_PORT:-8080}"
SERVER_LOG=/tmp/fish_server.log
rm -f "$SERVER_LOG"

# Localiza el modelo descargado
MODEL_SNAP=$(ls -d /root/.cache/huggingface/hub/models--fishaudio--s2-pro/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$MODEL_SNAP" ]; then
    echo "FATAL: model snapshot dir not found"
    exit 1
fi
echo "Model snapshot: $MODEL_SNAP"
echo "Files in snapshot:"
ls -la "$MODEL_SNAP" | head -20

# Arranca el servidor. Vemos qué argumentos acepta primero (--help).
echo
echo "--- api_server --help ---"
python -m tools.api_server --help 2>&1 | head -40 || true

echo
echo "--- starting api_server in background ---"
# Sin saber los argumentos exactos, intentamos lo más sensato basado
# en el patrón estándar de fish-speech: pasar --listen, --workers, y
# el path del modelo. Si falla, el log dirá qué falta.
python -m tools.api_server \
    --listen "0.0.0.0:${API_PORT}" \
    --llama-checkpoint-path "$MODEL_SNAP" \
    --decoder-checkpoint-path "$MODEL_SNAP" \
    --decoder-config-name modded_dac_vq \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Espera a que esté listo (probe HTTP)
echo "Waiting for server ready..."
for i in $(seq 1 60); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "FATAL: server died during boot. Last log lines:"
        tail -40 "$SERVER_LOG"
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${API_PORT}/" > /dev/null 2>&1 \
       || curl -sf "http://127.0.0.1:${API_PORT}/v1/models" > /dev/null 2>&1 \
       || curl -sf "http://127.0.0.1:${API_PORT}/health" > /dev/null 2>&1; then
        echo "Server up after ${i}s"
        break
    fi
    sleep 1
done

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "FATAL: server died waiting for ready. Log:"
    tail -60 "$SERVER_LOG"
    exit 1
fi

echo
echo "--- server log so far (head) ---"
head -50 "$SERVER_LOG"
echo
echo "--- discovering routes ---"
curl -s "http://127.0.0.1:${API_PORT}/openapi.json" 2>/dev/null \
    | python -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(list(d.get('paths',{}).keys()),indent=2))" \
    2>/dev/null \
    || echo "(no openapi.json available)"

echo
echo "--- POST /v1/tts test ---"
# Curl multipart con el ref + texto. Endpoint estándar fish-speech
# es /v1/tts; ajustaremos si openapi muestra otro nombre.
t0=$(date +%s.%N)
curl -sS -X POST "http://127.0.0.1:${API_PORT}/v1/tts" \
    -F "text=$TEST_TEXT" \
    -F "reference_audio=@/work/ref.wav" \
    -F "reference_text=$VOICE_TXT" \
    -o "/out/$OUT_NAME" \
    -w "HTTP %{http_code}, %{size_download} bytes\n" \
    || { echo "POST failed; last server log:"; tail -30 "$SERVER_LOG"; }
t1=$(date +%s.%N)
elapsed=$(echo "$t1 - $t0" | bc -l)

echo
echo "=== SMOKE TEST RESULT ==="
printf "Latency: %.1f s\n" "$elapsed"
if [ -f "/out/$OUT_NAME" ]; then
    ls -la "/out/$OUT_NAME"
fi

# Cleanup
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo
echo "--- final server log tail ---"
tail -30 "$SERVER_LOG"

echo
echo "=== smoke test done ==="
'

echo
echo "=== Host-side check ==="
if [ -f "$OUT_WAV" ]; then
    ls -lh "$OUT_WAV"
    echo "Reproduce con: ffplay $OUT_WAV   (o cualquier reproductor)"
else
    echo "WARNING: no output WAV at $OUT_WAV — revisa el log de arriba"
fi
