"""Re-cuantiza un GGUF de S2-Pro (fish-speech) desde F16 a un formato CUDA-nativo.

Motivo: el HF repo `rodrigomt/s2-pro-gguf` solo publica K-quants (q4_k_m,
q5_k_m, q6_k), q8_0 y f16. Las K-quants caen a CPU compute en CUDA porque
ggml v0.9.11 no implementa `get_rows` para superblocks K. q8_0 no entra en
6 GB VRAM. La única vía para GPU-full en una RTX 2060 es Q4_0 / Q4_1 / Q5_0
/ Q5_1, que sí están soportadas por CUDA `get_rows`.

Llama.cpp tiene `llama-quantize` pero rechaza la arquitectura `fish-speech`.
Este script usa `gguf` (lib oficial llama.cpp, agnóstica a la arquitectura
porque GGUF es un container genérico) para hacer el quantize tensor a tensor
sin tocar metadata específica del modelo.

Uso:

    python quantize.py INPUT.gguf OUTPUT.gguf --type q5_0

Por defecto preserva en F16 los tensores `*embeddings*` y los que terminan
en `.output.weight` o `.norm.*` para no degradar prosodia/calidad. Lo demás
se cuantiza al tipo solicitado.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# `gguf` viene como package independiente en PyPI mantenido por ggml-org.
# pip install gguf>=0.10
import gguf

log = logging.getLogger("quantize")


# ---------------------------------------------------------------------------
# Quantización Q4_0 y Q5_0
# ---------------------------------------------------------------------------
# Algoritmos públicos definidos en ggml/src/ggml-quants.c. Replicados aquí
# en numpy para poder cuantizar sin compilar ggml. Layouts de bloque:
#
#   Q4_0: 32 elementos por bloque. 1 scale (f16, 2 bytes) + 16 bytes de
#         nibbles (32 valores * 4 bits). Tamaño del bloque: 18 bytes.
#         Reconstrucción: x[i] = scale * (nibble[i] - 8).
#
#   Q5_0: 32 elementos por bloque. 1 scale (f16, 2 bytes) + 4 bytes de
#         high-bits packing + 16 bytes nibbles. Tamaño: 22 bytes.
#         Reconstrucción: x[i] = scale * (((high<<4)|nibble) - 16).
#
# Ambos usan absmax sobre el bloque para escoger el scale; nada de búsqueda
# de min — son simétricos en torno a 0. Para tensores TTS donde la
# distribución no está perfectamente centrada Q4_1/Q5_1 darían un punto más
# de calidad, pero a cambio cuestan +2 bytes/bloque y CUDA `get_rows` los
# soporta igual; añadir esos formatos es trivial si hace falta más adelante.
# ---------------------------------------------------------------------------

QK = 32  # block size (todas las quant Q*_0 / Q*_1 usan 32)


def _quantize_q4_0(data: np.ndarray) -> bytes:
    """Cuantiza un array float32 plano a bytes en formato Q4_0.

    El array debe tener longitud múltiplo de QK; el caller se encarga del
    padding/reshape antes de llamar.
    """
    assert data.size % QK == 0, "Q4_0 requiere n elementos múltiplo de 32"
    blocks = data.reshape(-1, QK).astype(np.float32)
    n_blocks = blocks.shape[0]

    # Scale por bloque: max(|x|)/-8 (rango Q4_0 = [-8, 7]).
    absmax = np.max(np.abs(blocks), axis=1)
    scale = np.where(absmax > 0, absmax / -8.0, 1.0).astype(np.float32)
    scale_inv = np.where(scale != 0, 1.0 / scale, 0.0)

    # Cuantiza a int [-8, 7] y suma 8 → unsigned [0, 15] que cabe en nibble.
    q = np.clip(np.round(blocks * scale_inv[:, None]) + 8, 0, 15).astype(np.uint8)

    # Empaqueta dos nibbles por byte (low en índice par, high en impar).
    low = q[:, :QK // 2]
    high = q[:, QK // 2:]
    packed = (low | (high << 4)).astype(np.uint8)  # (n_blocks, 16)

    out = bytearray(n_blocks * 18)
    scale_f16 = scale.astype(np.float16)
    for i in range(n_blocks):
        out[i * 18 : i * 18 + 2] = scale_f16[i].tobytes()
        out[i * 18 + 2 : i * 18 + 18] = packed[i].tobytes()
    return bytes(out)


def _quantize_q5_0(data: np.ndarray) -> bytes:
    """Q5_0: 5 bits por valor. Bit alto packed en un uint32 separado."""
    assert data.size % QK == 0, "Q5_0 requiere n elementos múltiplo de 32"
    blocks = data.reshape(-1, QK).astype(np.float32)
    n_blocks = blocks.shape[0]

    # Rango Q5_0 = [-16, 15] → scale = max(|x|)/-16.
    absmax = np.max(np.abs(blocks), axis=1)
    scale = np.where(absmax > 0, absmax / -16.0, 1.0).astype(np.float32)
    scale_inv = np.where(scale != 0, 1.0 / scale, 0.0)

    q = np.clip(np.round(blocks * scale_inv[:, None]) + 16, 0, 31).astype(np.uint8)
    # Bit 4 a `qh` (4 bytes uint32 little-endian); bits 0-3 a `qs` (16 bytes).
    high_bits = (q >> 4) & 1  # (n_blocks, 32)
    low_nibbles = q & 0x0F

    qh = np.zeros(n_blocks, dtype=np.uint32)
    for i in range(QK):
        qh |= (high_bits[:, i].astype(np.uint32) << i)

    low_packed = (low_nibbles[:, : QK // 2] | (low_nibbles[:, QK // 2 :] << 4)).astype(np.uint8)

    out = bytearray(n_blocks * 22)
    scale_f16 = scale.astype(np.float16)
    for i in range(n_blocks):
        base = i * 22
        out[base : base + 2] = scale_f16[i].tobytes()
        out[base + 2 : base + 6] = qh[i].tobytes()
        out[base + 6 : base + 22] = low_packed[i].tobytes()
    return bytes(out)


QUANT_FNS = {
    "q4_0": (_quantize_q4_0, gguf.GGMLQuantizationType.Q4_0),
    "q5_0": (_quantize_q5_0, gguf.GGMLQuantizationType.Q5_0),
}


# ---------------------------------------------------------------------------
# Política de qué tensores NO cuantizar
# ---------------------------------------------------------------------------
# Heurística estándar de llama.cpp: embeddings y output projection se
# preservan en F16 porque cuantizarlos degrada fuertemente la calidad de
# generación (especialmente prosodia en TTS). Para fish-speech los nombres
# de tensor difieren del esquema llama, así que matcheamos por substring.
KEEP_F16_PATTERNS = (
    "embeddings",   # token + codebook + fast embeddings
    "embed_tokens",
    ".output.",
    "output.weight",
    ".norm.",
    "_norm.",
    "norm.weight",
)


def _should_keep_f16(name: str) -> bool:
    return any(p in name for p in KEEP_F16_PATTERNS)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def quantize_gguf(src: Path, dst: Path, qtype: str) -> None:
    fn, ggml_type = QUANT_FNS[qtype]
    log.info("Reading %s", src)
    reader = gguf.GGUFReader(str(src), "r")

    # Detectar la arquitectura desde la metadata del fichero original para
    # inicializar el writer con el mismo valor (`gguf.GGUFWriter` lo exige
    # en el constructor).
    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        raise RuntimeError("missing general.architecture in input GGUF")
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
    log.info("Architecture: %s", arch)

    writer = gguf.GGUFWriter(str(dst), arch)

    # Copiar TODOS los KV pairs excepto los que el writer fija
    # automáticamente (general.architecture y general.alignment) y excepto
    # general.file_type, que reflejará el quant resultante.
    SKIP_KEYS = {"general.architecture", "general.alignment", "general.file_type"}
    for field in reader.fields.values():
        if field.name in SKIP_KEYS:
            continue
        # Cada field tiene el tipo guardado en `types[0]`; reusamos para
        # añadirlo con el mismo tipo. La rama list/array delega en
        # add_array para que el writer infiera el sub-tipo.
        ftype = field.types[0]
        if ftype == gguf.GGUFValueType.STRING:
            value = bytes(field.parts[field.data[0]]).decode("utf-8")
            writer.add_string(field.name, value)
        elif ftype == gguf.GGUFValueType.ARRAY:
            # Reconstruimos la lista python iterando los índices `data`.
            sub_type = field.types[1]
            values = []
            for idx in field.data:
                part = field.parts[idx]
                if sub_type == gguf.GGUFValueType.STRING:
                    values.append(bytes(part).decode("utf-8"))
                else:
                    values.append(part.tolist() if hasattr(part, "tolist") else part)
            writer.add_array(field.name, values)
        else:
            # Numérico simple → field.parts[field.data[0]] es un numpy scalar.
            value = field.parts[field.data[0]]
            if hasattr(value, "tolist"):
                value = value.tolist()
            # add_uint32 / add_float32 / etc. — usamos el helper genérico.
            writer.add_key_value(field.name, value, ftype)

    # `general.file_type` indica el quant principal (informativo, no usado
    # por el loader; pero llama.cpp lo escribe siempre).
    writer.add_uint32("general.file_type", int(_FILE_TYPE_FOR.get(qtype, 0)))

    # Plan: escribir headers de tensor primero (writer lo exige antes de
    # los datos), luego el payload de cada uno en orden.
    pending_data: list[tuple[str, bytes]] = []

    n_total = len(reader.tensors)
    n_quantized = 0
    n_kept = 0
    for i, tensor in enumerate(reader.tensors, 1):
        name = tensor.name
        shape = list(reversed(tensor.shape.tolist()))  # gguf almacena en orden invertido
        raw = tensor.data  # numpy view sobre el mmap
        src_type = tensor.tensor_type

        # Solo intentamos quantizar tensores f16/f32. Si ya está en otro
        # tipo (poco probable en un f16 puro pero por si acaso), lo dejamos
        # como está.
        is_floatish = src_type in (
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.F32,
        )
        keep_f16 = _should_keep_f16(name) or not is_floatish or raw.size % QK != 0

        if keep_f16:
            log.info("[%d/%d] keep   %-50s shape=%s type=%s",
                     i, n_total, name, shape, src_type.name)
            # Re-empaquetamos como F16 para uniformidad (si venía f32 lo
            # reducimos; si ya era f16 sale igual).
            arr = np.asarray(raw, dtype=np.float16 if is_floatish else None)
            writer.add_tensor(name, arr, raw_dtype=None)
            n_kept += 1
            continue

        log.info("[%d/%d] quant  %-50s shape=%s -> %s",
                 i, n_total, name, shape, qtype.upper())
        flat = np.asarray(raw, dtype=np.float32).ravel()
        packed = fn(flat)
        # `add_tensor` con `raw_shape` + `raw_dtype` permite inyectar bytes
        # ya cuantizados sin que el writer reinterprete.
        np_packed = np.frombuffer(packed, dtype=np.uint8)
        writer.add_tensor(
            name,
            np_packed,
            raw_shape=tuple(shape),
            raw_dtype=ggml_type,
        )
        n_quantized += 1

    log.info("Writing header...")
    writer.write_header_to_file()
    log.info("Writing KVs...")
    writer.write_kv_data_to_file()
    log.info("Writing tensors...")
    writer.write_tensors_to_file()
    writer.close()

    log.info("Done. Quantized=%d, kept=%d, total=%d", n_quantized, n_kept, n_total)
    log.info("Output: %s (%.1f MB)", dst, dst.stat().st_size / (1024 * 1024))


# Ver llama.cpp/include/llama.h para los IDs canónicos de file_type.
_FILE_TYPE_FOR = {
    "q4_0": 2,
    "q5_0": 8,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path al GGUF f16 origen")
    parser.add_argument("output", type=Path, help="Path al GGUF cuantizado destino")
    parser.add_argument("--type", choices=sorted(QUANT_FNS), default="q5_0",
                        help="Tipo de cuantización (default: q5_0)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.input.exists():
        log.error("Input not found: %s", args.input)
        return 1
    if args.output.exists():
        log.error("Output already exists, refusing to overwrite: %s", args.output)
        return 1

    quantize_gguf(args.input, args.output, args.type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
