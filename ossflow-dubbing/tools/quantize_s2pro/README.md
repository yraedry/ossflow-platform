# quantize_s2pro

Re-cuantiza un GGUF de S2-Pro (`fish-speech` arch) desde F16 a un formato
CUDA-nativo (Q4_0 / Q5_0). El repo HF oficial `rodrigomt/s2-pro-gguf` solo
publica K-quants y q8_0; las K-quants caen a CPU compute en ggml v0.9.11
(usado por s2.cpp@e48ce8e), y q8_0 no entra en una RTX 2060 6 GB.

`llama-quantize` no sirve aquí porque rechaza la arquitectura `fish-speech`.
Este script usa `gguf-py` para hacer el quantize tensor a tensor sin tocar
metadata específica del modelo.

## Uso (oneshot, en el LXC del server)

Asume que el f16 ya está descargado en
`/opt/ossflow/ossflow-platform/models/s2pro-gguf/s2-pro-f16.gguf`.

```bash
cd /opt/ossflow/ossflow-platform

docker run --rm \
  -v "$(pwd)/models/s2pro-gguf:/models" \
  -v "$(pwd)/ossflow-dubbing/tools/quantize_s2pro:/tool:ro" \
  python:3.11-slim bash -c '
    pip install --no-cache-dir "gguf>=0.10" numpy && \
    python /tool/quantize.py /models/s2-pro-f16.gguf /models/s2-pro-q5_0.gguf --type q5_0
  '
```

Tras la primera ejecución, recarga la UI de Settings: el desplegable de
cuantización (poblado en caliente vía `GET /api/dubbing/s2pro/models`)
listará el nuevo `q5_0`. Selecciónalo, guarda, lanza un dub y comprueba
con `nvidia-smi -l 1` que el GPU sale del estado P8 0% mientras corre.

## Tipos soportados

Todos vienen del backend `gguf.quants` oficial de llama.cpp; el GGUF
resultante es binariamente idéntico al que produciría `llama-quantize`.

| `--type` | Bytes/bloque | Tamaño aprox | CUDA `get_rows` | Uso |
|----------|--------------|--------------|-----------------|-----|
| `q4_0`   | 18 (32 vals) | ~3.0 GB      | ✅              | Mínimo VRAM |
| `q4_1`   | 20 (32 vals) | ~3.3 GB      | ✅              | +1 byte por bloque vs q4_0 (asimétrico) |
| `q5_0`   | 22 (32 vals) | ~4.0 GB      | ✅              | Balance recomendado para 2060 6 GB |
| `q5_1`   | 24 (32 vals) | ~4.3 GB      | ✅              | +1 byte por bloque vs q5_0 (asimétrico) |
| `q8_0`   | 34 (32 vals) | ~5.6 GB      | ✅              | Calidad máxima cuantizada — OOM probable en 2060 |

## Heurística de tensores preservados en F16

Para no degradar prosodia, los siguientes tensores NO se cuantizan
(matcheo por substring sobre el nombre):

* `*embeddings*` — token, codebook y fast embeddings.
* `*.output.*`, `output.weight` — proyección final.
* `*.norm.*`, `*_norm.*`, `norm.weight` — RMSNorm scales.

El resto (atención QKV, FFN, etc.) sí se cuantiza. Es la misma política
que aplica `llama-quantize` por defecto a modelos llama/qwen.
