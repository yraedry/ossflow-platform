"""Predicted MOS (mean opinion score) for a dubbed audio track.

Uses UTMOS22 (sarulab-speech), a neural network trained on crowdsourced
MOS ratings of TTS samples. Outputs one number 1-5 that correlates with
perceived naturalness — we use it as the global quality signal that the
boundary report cannot produce.

Two design constraints:

* **Lazy load.** UTMOS pulls torchaudio + a ~100 MB checkpoint on first
  use. Loading at import time would slow every pipeline start-up and
  import failures would kill dubbing. Instead ``score_mos`` loads on
  demand and caches the model in a module-level singleton.
* **Best effort.** If the UTMOS wheel, its weights or GPU are missing,
  ``score_mos`` returns ``None``. The caller writes a blank MOS field
  into the QA sidecar and the pipeline continues.

Licence note: UTMOS22 is distributed under CC BY-NC 4.0 (non-commercial).
This codebase is for personal/internal use only; swap to DNSMOS (MIT) if
that ever changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Module-level singleton so we load the model once per process.
_MODEL = None
_MODEL_FAILED = False


@dataclass
class MosScore:
    """Neural MOS prediction over a full audio file."""

    score: float                # 1.0 – 5.0
    model_name: str             # "utmos22_strong"
    sample_rate_used: int

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def score_mos(wav_path: Path) -> Optional[MosScore]:
    """Predict MOS for a wav file. Returns None if UTMOS isn't available."""
    global _MODEL_FAILED

    if _MODEL_FAILED:
        return None

    model = _get_model()
    if model is None:
        return None

    try:
        import torch
        import torchaudio
    except ImportError:
        logger.warning("torch/torchaudio missing; skipping MOS")
        _MODEL_FAILED = True
        return None

    try:
        wave, sr = torchaudio.load(str(wav_path))
        # UTMOS22 was trained on 16 kHz mono input. Downsampling is standard.
        target_sr = 16000
        if sr != target_sr:
            wave = torchaudio.functional.resample(wave, sr, target_sr)
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)

        device = next(model.parameters()).device
        wave = wave.to(device)

        with torch.no_grad():
            score = model(wave, target_sr)

        score_f = float(score.detach().cpu().item())
        # Clamp to the valid MOS range — UTMOS can spit slightly out-of-range
        # values on noisy clips (training-distribution mismatch).
        score_f = max(1.0, min(5.0, score_f))

        return MosScore(
            score=round(score_f, 2),
            model_name="utmos22_strong",
            sample_rate_used=target_sr,
        )
    except Exception as exc:
        logger.warning("MOS inference failed on %s: %s", wav_path.name, exc)
        return None


# ----------------------------------------------------------------------
# Model loader (lazy, cached)
# ----------------------------------------------------------------------

def _get_model():
    global _MODEL, _MODEL_FAILED
    if _MODEL is not None:
        return _MODEL
    if _MODEL_FAILED:
        return None

    try:
        import torch
    except ImportError:
        logger.warning("torch unavailable; MOS scoring disabled")
        _MODEL_FAILED = True
        return None

    try:
        # UTMOS is served via torch.hub from tarepan/SpeechMOS. Single entry
        # point, no repo clone needed — torch.hub caches under ~/.cache/torch.
        # trust_repo=True is required because we're pulling third-party code.
        _MODEL = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
        _MODEL.eval()
        # CPU-only by design. UTMOS is a tiny model (~100 MB) and runs in ~5s
        # per clip on CPU. On 6 GB cards (RTX 2060) the GPU is already saturated
        # by Demucs/S2-Pro/voice cloner during a dubbing job — pinning UTMOS to
        # CUDA causes OOM on the QA fase. CPU keeps QA strictly best-effort.
        logger.info("UTMOS22 loaded (device=%s)", next(_MODEL.parameters()).device)
        return _MODEL
    except Exception as exc:
        logger.warning(
            "UTMOS22 failed to load (%s) — MOS scoring disabled", exc,
        )
        _MODEL_FAILED = True
        return None
