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
VOICE_WAV="${VOICE_WAV:-${VOICES_DIR}/craig_10s.wav}"
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
apt-get install -y --no-install-recommends -qq \
    python3.11 python3.11-venv python3-pip git ffmpeg \
    build-essential portaudio19-dev libsndfile1 \
    > /dev/null 2>&1
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 > /dev/null

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
python -c "
import torch
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(f\"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}\")
print(f\"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\" if torch.cuda.is_available() else \"\")
"

echo "--- Stage 6: synthesize test phrase ---"
# fish-speech expone varios entrypoints. El más estable para voice
# clone vía CLI es `tools/llama/generate.py` para tokens semánticos +
# `tools/vqgan/inference.py` para decode. Pero hay un wrapper más
# nuevo en `tools/api_server.py` que sirve HTTP. Para este smoke test
# uso el script python directo más simple.
python - <<PYEOF
import time, sys
sys.path.insert(0, "/work/fish-speech")

# Importar el inference helper top-level. Si la API ha cambiado en
# main, ajustaremos en el siguiente intento.
try:
    from tools.api_server import inference_engine
except ImportError as e:
    print(f"NOTE: tools.api_server import failed ({e})")
    print("Trying fish_speech.inference instead...")
    try:
        from fish_speech.models.text2semantic.inference import generate_long
        from fish_speech.models.vqgan.inference import wav_inference
    except ImportError as e2:
        print(f"FATAL: cannot find inference API: {e2}")
        sys.exit(2)

import os
ref_wav = "/work/ref.wav"
ref_txt = os.environ["VOICE_TXT"]
text = os.environ["TEST_TEXT"]
out_path = f"/out/{os.environ[\"OUT_NAME\"]}"

print(f"Reference: {ref_wav}")
print(f"Reference text: {ref_txt[:80]}...")
print(f"Text to synthesize: {text}")
print(f"Output: {out_path}")

t0 = time.time()
# Aquí va la llamada real — el código exacto depende de la API
# disponible (api_server vs scripts directos). Lo metemos en un bloque
# que catch cualquier diferencia de firma y reporte qué hay disponible.
try:
    # Most likely path: api_server expone una función inference()
    # que acepta text + reference_audio + reference_text y devuelve WAV bytes.
    result = inference_engine.inference(
        text=text,
        reference_audio=ref_wav,
        reference_text=ref_txt,
    )
    with open(out_path, "wb") as f:
        f.write(result)
except Exception as e:
    print(f"FATAL during inference: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    sys.exit(3)

elapsed = time.time() - t0
print(f"\n=== SMOKE TEST RESULT ===")
print(f"Latency: {elapsed:.1f}s")
print(f"Output WAV: {out_path}")
import os.path
print(f"Output size: {os.path.getsize(out_path)/1024:.1f} KB")

# VRAM final
import torch
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM allocated: {used:.2f} GB")
    print(f"VRAM reserved:  {reserved:.2f} GB")
PYEOF

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
