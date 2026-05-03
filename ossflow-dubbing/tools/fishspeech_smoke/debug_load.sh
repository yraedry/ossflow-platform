#!/usr/bin/env bash
# Debug por qué el modelo se cuelga en "Loading model from ..." sin
# subir VRAM ni morir. Lanza el server en foreground con TODOS los
# loggers torch/safetensors/loguru a DEBUG, y attachea py-spy al PID
# tras 30s para sacar el stack del thread atascado.

set -euo pipefail

VOICES_DIR="${VOICES_DIR:-/opt/ossflow/ossflow-platform/ossflow-dubbing/voices}"
VOICE_WAV="${VOICE_WAV:-${VOICES_DIR}/craig_16s.wav}"
MODELS_DIR="${MODELS_DIR:-/opt/ossflow/ossflow-platform/models/fish-speech-cache}"
PIP_CACHE_DIR_HOST="${MODELS_DIR%/*}/fishspeech-pip-cache"
FISH_REPO_DIR_HOST="${MODELS_DIR%/*}/fish-speech-src"

mkdir -p "$PIP_CACHE_DIR_HOST" "$FISH_REPO_DIR_HOST"

docker run --rm -it \
    --runtime=nvidia \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$MODELS_DIR:/root/.cache/huggingface" \
    -v "$PIP_CACHE_DIR_HOST:/root/.cache/pip" \
    -v "$FISH_REPO_DIR_HOST:/work/fish-speech" \
    -v "$VOICE_WAV:/work/ref.wav:ro" \
    nvidia/cuda:12.4.1-runtime-ubuntu22.04 bash -c '
set -euo pipefail

# Asume que /work/fish-speech YA está cacheado del smoke_test previo
# (mismo bind-mount). Si no está, lo clonamos. Igual con deps Python:
# si torch no está importable, instalamos todo. Detectamos vía rc.
export DEBIAN_FRONTEND=noninteractive
if ! command -v python3 >/dev/null 2>&1 || ! command -v pip3 >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends -qq \
        python3 python3-dev python3-pip git ffmpeg \
        build-essential portaudio19-dev libsndfile1 \
        > /dev/null 2>&1
    ln -sf /usr/bin/python3 /usr/local/bin/python
    ln -sf /usr/bin/pip3 /usr/local/bin/pip
fi
# Si fish_speech no es importable, asumimos que el smoke nunca lo dejó
# instalado y lo metemos ahora (pip cache lo agiliza).
if ! python -c "import fish_speech" 2>/dev/null; then
    echo "--- installing torch + fish-speech (first time, ~5min) ---"
    if [ ! -d /work/fish-speech/.git ]; then
        git clone --depth 50 https://github.com/fishaudio/fish-speech.git /work/fish-speech
    fi
    cd /work/fish-speech
    pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124 > /tmp/pip.log 2>&1 \
      || { echo "FAIL torch"; tail -30 /tmp/pip.log; exit 1; }
    # En lugar de instalar el paquete, instalamos sus deps + usamos
    # PYTHONPATH para imports. PEP 660 (editable installs) lo rompe en
    # el pyproject de fish-speech main; install no-editable mete el
    # paquete en site-packages y los `python -m tools.api_server`
    # quedan inconsistentes con el clone.
    if [ -f pyproject.toml ]; then
        # Saca deps de pyproject (manera limpia: pip install . falla;
        # pero `pip install --dry-run` resuelve y muestra. Usamos un
        # truco: pip install la versión publicada del paquete (si
        # existe) trae las mismas deps sin tocar nuestro source local.
        pip install --no-cache-dir fish-speech > /tmp/pip.log 2>&1 || true
        # Si fish-speech no está en PyPI, instalamos las deps comunes
        # explícitamente. Todas vienen ya con torch o son ligeras.
        pip install --no-cache-dir \
            transformers tokenizers accelerate einops loguru pyrootutils \
            "kui[asgi]" uvicorn[standard] httpx multipart msgpack \
            librosa soundfile numpy hydra-core omegaconf \
            descript-audio-codec descript-audiotools \
            > /tmp/pip2.log 2>&1 \
          || { echo "FAIL deps"; tail -30 /tmp/pip2.log; exit 1; }
    fi
fi
export PYTHONPATH=/work/fish-speech:${PYTHONPATH:-}
cd /work/fish-speech

# py-spy para profile en vivo
pip install --quiet py-spy 2>&1 | tail -2

MODEL_SNAP=$(ls -d /root/.cache/huggingface/hub/models--fishaudio--s2-pro/snapshots/*/ | head -1)
echo "Snapshot: $MODEL_SNAP"

# Verifica que los blobs son leíbles (los .safetensors son symlinks)
echo "--- check files ---"
file "$MODEL_SNAP/codec.pth" "$MODEL_SNAP/model-00001-of-00002.safetensors" 2>&1 | head -5
ls -laL "$MODEL_SNAP/codec.pth" "$MODEL_SNAP/model-00001-of-00002.safetensors" 2>&1 | head -5

# Test 1: ¿pytorch puede leer el codec.pth standalone?
echo
echo "--- Test 1: torch.load(codec.pth) standalone ---"
timeout 60 python -c "
import torch, time
t0 = time.time()
print(\"loading codec.pth...\", flush=True)
sd = torch.load(\"$MODEL_SNAP/codec.pth\", map_location=\"cpu\")
print(f\"OK in {time.time()-t0:.1f}s, keys={len(sd) if isinstance(sd, dict) else type(sd).__name__}\", flush=True)
" 2>&1 | tail -10
echo "Exit: $?"

# Test 2: ¿safetensors puede leer model-00001?
echo
echo "--- Test 2: safetensors load ---"
timeout 60 python -c "
import time
from safetensors import safe_open
t0 = time.time()
print(\"opening safetensors...\", flush=True)
with safe_open(\"$MODEL_SNAP/model-00001-of-00002.safetensors\", framework=\"pt\", device=\"cpu\") as f:
    keys = list(f.keys())
print(f\"OK in {time.time()-t0:.1f}s, n_tensors={len(keys)}\", flush=True)
" 2>&1 | tail -10
echo "Exit: $?"

# Test 3: cargar el modelo Llama directamente con la API de fish_speech
echo
echo "--- Test 3: fish_speech model load with py-spy attached ---"
python -c "
import sys, time, torch
sys.path.insert(0, \"/work/fish-speech\")
print(\"loading DualARTransformer.from_pretrained...\", flush=True)
from fish_speech.models.text2semantic.llama import DualARTransformer
t0 = time.time()
model = DualARTransformer.from_pretrained(
    \"$MODEL_SNAP\",
    load_weights=True,
    max_length=2048,
)
print(f\"loaded in {time.time()-t0:.1f}s\", flush=True)
print(f\"moving to cuda+bf16...\", flush=True)
t0 = time.time()
model = model.to(device=\"cuda\", dtype=torch.bfloat16)
print(f\"moved in {time.time()-t0:.1f}s\", flush=True)
print(f\"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB\", flush=True)
" 2>&1 &
PYPID=$!
echo "Python PID: $PYPID"
sleep 60
echo
echo "--- py-spy dump after 60s ---"
py-spy dump --pid $PYPID 2>&1 | head -40 || echo "(py-spy failed; process may have ended)"
echo
echo "--- waiting for python to finish or 60s more ---"
( sleep 60; kill $PYPID 2>/dev/null ) &
wait $PYPID 2>/dev/null || true
'
