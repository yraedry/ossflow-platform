"""Normalización RMS de segmentos TTS.

Migrado de ``pipeline.py:_normalize_tts_levels`` en T32.2. Función
pura — recibe la lista de segmentos y los modifica in-place. No
depende de la configuración del pipeline ni de otros métodos de
``DubbingPipeline``.

Estrategia:

1. Per-segmento: aplica gain hacia ``TARGET_RMS_DBFS`` con caps
   ``MAX_BOOST`` / ``MAX_CUT`` y peak-guard a ``PEAK_CEIL_DBFS``.
2. Pair-pass: detecta saltos RMS entre vecinos > ``PAIR_JUMP_TRIGGER_DB``
   y eleva el segmento más bajo (nunca baja el alto — el ducking
   downstream necesita cuerpo en la voz).
"""

from __future__ import annotations

import logging
import math

from dubbing_generator.audio.mixer import TtsSegment

logger = logging.getLogger(__name__)


# S2-Pro tuning (RMS distribution observada en Craig Jones S04E05).
TARGET_RMS_DBFS = -22.0
MAX_BOOST = 14.0
MAX_CUT = 12.0
GATE_THRESHOLD_DBFS = -40.0
PEAK_CEIL_DBFS = -1.0
SILENCE_FLOOR_DBFS = -55.0
PAIR_JUMP_TRIGGER_DB = 4.0
PAIR_LIFT_MAX_DB = 7.0


def _gated_rms_dbfs(seg) -> float:
    """Gated RMS in dBFS — ignora samples bajo ``GATE_THRESHOLD_DBFS``.

    Sin gate, el silencio residual tras trim baja artificialmente
    la RMS global y el gain se sobreajusta. El gate restringe la
    medida a la parte vocalizada de la frase.
    """
    samples = seg.get_array_of_samples()
    if len(samples) == 0:
        return float("-inf")
    max_val = float(1 << (8 * seg.sample_width - 1))
    gate = 10.0 ** (GATE_THRESHOLD_DBFS / 20.0)
    sq_sum = 0.0
    count = 0
    for v in samples:
        mag = abs(v) / max_val
        if mag >= gate:
            sq_sum += mag * mag
            count += 1
    # Fallback: si todo está bajo el gate, medimos sin gate para no -inf.
    if count == 0:
        count = len(samples)
        sq_sum = sum((v / max_val) ** 2 for v in samples)
    if sq_sum <= 0:
        return float("-inf")
    rms = math.sqrt(sq_sum / count)
    return 20.0 * math.log10(rms) if rms > 0 else float("-inf")


def normalize_tts_levels(segments: list[TtsSegment]) -> None:
    """Iguala loudness entre frases TTS por RMS (no por peak).

    Pipeline en dos pasadas:

    1. Per-segment pass — converge cada segmento hacia ``TARGET_RMS_DBFS``
       con caps de gain (``MAX_BOOST`` / ``MAX_CUT``) y peak-guard a
       ``PEAK_CEIL_DBFS``. Skipea silent segments (< ``SILENCE_FLOOR_DBFS``)
       para no amplificar ruido.
    2. Pair-pass — detecta saltos > ``PAIR_JUMP_TRIGGER_DB`` entre vecinos
       y nudge al lower (nunca al higher — ducking downstream necesita
       cuerpo en la voz). Marca ``rms_jump_boundary=True`` en ambos
       vecinos para que el mixer use crossfade más fuerte (rms_xfade).
    """
    # ─── Per-segment pass ───────────────────────────────────────────
    for seg in segments:
        if seg.audio is None or len(seg.audio) == 0:
            continue
        cur_rms = _gated_rms_dbfs(seg.audio)
        if not math.isfinite(cur_rms):
            continue
        if cur_rms < SILENCE_FLOOR_DBFS:
            logger.debug(
                "Skipping level normalization on silent segment "
                "(gated RMS %.1f dBFS)", cur_rms,
            )
            continue
        delta = TARGET_RMS_DBFS - cur_rms
        if delta > MAX_BOOST:
            delta = MAX_BOOST
        elif delta < -MAX_CUT:
            delta = -MAX_CUT
        if abs(delta) >= 0.3:
            seg.audio = seg.audio.apply_gain(delta)
        # Peak safety: solo bajamos si el pico se sale del headroom.
        cur_peak = seg.audio.max_dBFS
        if math.isfinite(cur_peak) and cur_peak > PEAK_CEIL_DBFS:
            seg.audio = seg.audio.apply_gain(PEAK_CEIL_DBFS - cur_peak)

    # ─── Pair-pass: pairwise RMS leveling ───────────────────────────
    sorted_segs = sorted(
        (s for s in segments if s.audio is not None and len(s.audio) > 0),
        key=lambda s: s.start_ms,
    )
    for i in range(len(sorted_segs) - 1):
        a, b = sorted_segs[i], sorted_segs[i + 1]
        rms_a = _gated_rms_dbfs(a.audio)
        rms_b = _gated_rms_dbfs(b.audio)
        if not (math.isfinite(rms_a) and math.isfinite(rms_b)):
            continue
        diff = rms_a - rms_b
        if abs(diff) < PAIR_JUMP_TRIGGER_DB:
            continue
        # Solo elevamos el lado más bajo. Cap el lift y respetamos
        # peak headroom.
        lift = min(PAIR_LIFT_MAX_DB, abs(diff) - PAIR_JUMP_TRIGGER_DB + 2.0)
        victim = b if diff > 0 else a
        new_audio = victim.audio.apply_gain(lift)
        cur_peak = new_audio.max_dBFS
        if math.isfinite(cur_peak) and cur_peak > PEAK_CEIL_DBFS:
            new_audio = new_audio.apply_gain(PEAK_CEIL_DBFS - cur_peak)
        victim.audio = new_audio
        # Flag ambos vecinos para que el mixer use rms_xfade (longer).
        a.rms_jump_boundary = True
        b.rms_jump_boundary = True
        logger.debug(
            "pair-lift boundary %d: diff=%.1f dB → lift %.1f dB on %s side",
            i, diff, lift, "next" if diff > 0 else "prev",
        )
