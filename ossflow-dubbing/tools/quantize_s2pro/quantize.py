"""Re-cuantiza un GGUF de S2-Pro (fish-speech) desde F16 a un formato CUDA-nativo.

Motivo: el HF repo `rodrigomt/s2-pro-gguf` solo publica K-quants (q4_k_m,
q5_k_m, q6_k), q8_0 y f16. Las K-quants caen a CPU compute en CUDA porque
ggml v0.9.11 no implementa `get_rows` para superblocks K. q8_0 no entra en
6 GB VRAM. La única vía para GPU-full en una RTX 2060 es Q4_0 / Q4_1 /
Q5_0 / Q5_1 / Q8_0, soportadas por CUDA `get_rows`.

`llama-quantize` no sirve aquí porque rechaza la arquitectura `fish-speech`.
Usamos `gguf-py` (lib oficial llama.cpp, agnóstica a la arquitectura
porque GGUF es un container genérico) para hacer el quantize tensor a
tensor sin tocar metadata específica del modelo. La función
`gguf.quants.quantize()` implementa los layouts canónicos de
`ggml/src/ggml-quants.c` en numpy puro, así que el GGUF resultante es
binariamente idéntico al que produciría `llama-quantize`.

Uso:

    python quantize.py INPUT.gguf OUTPUT.gguf --type q5_0

Por defecto preserva en F16 los tensores `*embeddings*`, los `*.output.*`
y los `*.norm.*` para no degradar prosodia. Lo demás se cuantiza al tipo
solicitado. Es la misma política que aplica `llama-quantize` por defecto.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

import gguf
from gguf.quants import quantize as gguf_quantize

log = logging.getLogger("quantize")


# `general.file_type` ID canónico (ver llama.cpp/include/llama.h enum
# llama_ftype). Lo escribimos sólo para que herramientas externas que
# inspeccionen el GGUF sepan qué quant predomina; el loader de s2.cpp se
# basa en los tensor types directamente, no en este campo.
FILE_TYPE_FOR = {
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 8,
    "q5_1": 9,
    "q8_0": 7,
}

QUANT_TYPE_FOR = {
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
    "q4_1": gguf.GGMLQuantizationType.Q4_1,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q5_1": gguf.GGMLQuantizationType.Q5_1,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}


# Heurística estándar de `llama-quantize`: embeddings y output projection
# se preservan en F16 porque cuantizarlos degrada fuertemente la calidad
# de generación (especialmente prosodia en TTS). Para fish-speech los
# nombres de tensor difieren del esquema llama, así que matcheamos por
# substring. Norms son tan pequeños (vector 1D) que no merece la pena
# cuantizarlos y degradan mucho si se hace.
KEEP_F16_PATTERNS = (
    "embeddings",       # token + codebook + fast embeddings (fish-speech)
    "embed_tokens",     # alias estándar llama
    ".output.",
    "output.weight",
    ".norm.",
    "_norm.",
    "norm.weight",
)

# El codec (encoder/decoder de audio) ejecuta en CPU dentro de s2.cpp y
# solo soporta tensores F16/F32 — cuantizarlos rompe el load con
# "[Codec] unsupported tensor type for host copy". El propio HF README
# de rodrigomt/s2-pro-gguf indica que las quants oficiales mantienen
# `c.*` en F16. En fish-speech todos los tensores del codec usan el
# prefijo `c.` (codec), así que matcheamos por prefijo en lugar de
# substring para no falsear con nombres que casualmente lo contengan.
KEEP_F16_PREFIXES = (
    "c.",  # codec encoder + decoder + RVQ quantizer + transformer interno
)


def _should_keep_f16(name: str) -> bool:
    if any(name.startswith(p) for p in KEEP_F16_PREFIXES):
        return True
    return any(p in name for p in KEEP_F16_PATTERNS)


_INT_TYPES = {
    gguf.GGUFValueType.UINT8, gguf.GGUFValueType.INT8,
    gguf.GGUFValueType.UINT16, gguf.GGUFValueType.INT16,
    gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32,
    gguf.GGUFValueType.UINT64, gguf.GGUFValueType.INT64,
}
_FLOAT_TYPES = {gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64}


def _coerce_scalar(value, ftype: gguf.GGUFValueType):
    """Convierte un escalar (numpy o python) al tipo Python que `struct.pack` espera.

    `field.parts[idx]` viene como numpy scalar; `.tolist()` lo pasa a
    int/float/bool nativo. Pero algunos enteros se quedan como float si
    el dtype subyacente es float64 (raro, pero ocurre con campos
    `general.alignment` o ciertos legacy GGUF) — y `struct.pack('I', val)`
    rechaza floats. Forzamos el tipo según lo que dice el field type.
    """
    raw = value.tolist() if hasattr(value, "tolist") else value
    # `field.parts[idx]` puede ser un numpy array de 1 elemento en lugar
    # de un escalar; en ese caso `.tolist()` devuelve [v] y necesitamos
    # extraer el primero antes de aplicar int()/float()/bool().
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(f"expected scalar, got list len={len(raw)}: {raw!r}")
        raw = raw[0]
    if ftype in _INT_TYPES:
        return int(raw)
    if ftype in _FLOAT_TYPES:
        return float(raw)
    if ftype == gguf.GGUFValueType.BOOL:
        return bool(raw)
    return raw


def _copy_kv(reader: gguf.GGUFReader, writer: gguf.GGUFWriter,
             skip_keys: set[str]) -> None:
    """Copia todos los KV pairs del reader al writer salvo los excluidos."""
    for field in reader.fields.values():
        if field.name in skip_keys:
            continue
        ftype = field.types[0]
        if ftype == gguf.GGUFValueType.STRING:
            value = bytes(field.parts[field.data[0]]).decode("utf-8")
            writer.add_string(field.name, value)
        elif ftype == gguf.GGUFValueType.ARRAY:
            sub_type = field.types[1]
            values: list = []
            for idx in field.data:
                part = field.parts[idx]
                if sub_type == gguf.GGUFValueType.STRING:
                    values.append(bytes(part).decode("utf-8"))
                else:
                    values.append(_coerce_scalar(part, sub_type))
            writer.add_array(field.name, values)
        else:
            value = _coerce_scalar(field.parts[field.data[0]], ftype)
            writer.add_key_value(field.name, value, ftype)


def quantize_gguf(src: Path, dst: Path, qtype_name: str) -> None:
    qtype = QUANT_TYPE_FOR[qtype_name]

    log.info("Reading %s", src)
    reader = gguf.GGUFReader(str(src), "r")

    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        raise RuntimeError("missing general.architecture in input GGUF")
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
    log.info("Architecture: %s", arch)

    writer = gguf.GGUFWriter(str(dst), arch)

    # general.architecture y general.alignment los fija el constructor del
    # writer; general.file_type lo reescribimos al final con el quant
    # resultante.
    _copy_kv(reader, writer, skip_keys={
        "general.architecture",
        "general.alignment",
        "general.file_type",
    })
    writer.add_uint32("general.file_type", FILE_TYPE_FOR.get(qtype_name, 0))

    n_total = len(reader.tensors)
    n_quantized = 0
    n_kept = 0

    for i, tensor in enumerate(reader.tensors, 1):
        name = tensor.name
        # gguf-py expone shape en orden invertido respecto al GGUF en
        # disco; revertimos para que (rows, cols) lea natural.
        shape = list(reversed(tensor.shape.tolist()))
        src_type = tensor.tensor_type
        raw = tensor.data  # numpy view sobre el mmap

        is_floatish = src_type in (
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.F32,
        )
        # `gguf.quants.quantize` exige que la última dimensión sea múltiplo
        # del block size (32 para Q*_0/Q*_1, 32 también para Q8_0). Si no
        # cuadra, dejamos en F16: sería un tensor exótico; preservar es
        # más seguro que romper el binario.
        block_size = gguf.GGML_QUANT_SIZES[qtype][0]
        last_dim_ok = shape[-1] % block_size == 0
        keep_f16 = (
            _should_keep_f16(name)
            or not is_floatish
            or not last_dim_ok
        )

        if keep_f16:
            log.info("[%d/%d] keep   %-50s shape=%s type=%s",
                     i, n_total, name, shape, src_type.name)
            arr = np.asarray(raw, dtype=np.float16) if is_floatish else np.asarray(raw)
            writer.add_tensor(name, arr)
            n_kept += 1
            continue

        log.info("[%d/%d] quant  %-50s shape=%s -> %s",
                 i, n_total, name, shape, qtype_name.upper())
        # gguf.quants.quantize() devuelve un array uint8 con el SHAPE EN
        # BYTES ya aplicado (ver __shape_to_bytes en gguf/quants.py).
        # add_tensor() detecta dtype=uint8 + raw_dtype=Q* y revierte el
        # shape de bytes a shape de elementos vía
        # quant_shape_from_byte_shape — así que NO debemos pasar
        # raw_shape; si lo pasamos en elementos, el writer lo trata como
        # bytes y rompe con "X is not a multiple of type size Y".
        arr_f32 = np.asarray(raw, dtype=np.float32).reshape(shape)
        packed = gguf_quantize(arr_f32, qtype)
        writer.add_tensor(name, packed, raw_dtype=qtype)
        n_quantized += 1

    log.info("Writing header...")
    writer.write_header_to_file()
    log.info("Writing KVs...")
    writer.write_kv_data_to_file()
    log.info("Writing tensors...")
    writer.write_tensors_to_file()
    writer.close()

    size_mb = dst.stat().st_size / (1024 * 1024)
    log.info("Done. Quantized=%d, kept=%d, total=%d", n_quantized, n_kept, n_total)
    log.info("Output: %s (%.1f MB)", dst, size_mb)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path al GGUF f16 origen")
    parser.add_argument("output", type=Path, help="Path al GGUF cuantizado destino")
    parser.add_argument("--type", choices=sorted(QUANT_TYPE_FOR), default="q5_0",
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
