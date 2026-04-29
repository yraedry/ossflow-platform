"""Per-boundary objective metrics for a dubbed audio track.

The dubbing mixer places N TTS segments on a timeline. Between every pair
of adjacent segments there is a *boundary* — the point where one phrase
ends and the next begins. Listeners perceive a boundary as "chirriante"
when any of these change abruptly:

* energy (RMS jump)
* timbre (spectral-centroid jump)
* pitch (F0 jump)
* temporal gap (too tight → clipping; too loose → dead air)

This module computes those four metrics per boundary and classifies each
boundary as ``ok`` / ``warn`` / ``hard``. The report is a flat dict that
serialises straight to the ``.dub-qa.json`` sidecar so the frontend can
show an overlay on the timeline.

No models, no GPU — numpy + librosa only. Fast enough to run inline in
the dubbing pipeline (<1 s per 40-min lecture).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Thresholds — tuned on XTTS-v2 BJJ dubs (single-speaker coach).
#
# XTTS renders each phrase independently, so even after global loudness
# normalization some natural variation (~3-5 dB RMS, ~500-700 Hz
# centroid) survives between consecutive phrases. Older thresholds
# (6 dB hard, 800 Hz hard) flagged every boundary as hard; the listener
# barely perceives jumps until ~10 dB RMS or ~1200 Hz centroid.
# ----------------------------------------------------------------------

_RMS_JUMP_WARN_DB = 5.0
_RMS_JUMP_HARD_DB = 10.0
_CENTROID_JUMP_WARN_HZ = 700.0
_CENTROID_JUMP_HARD_HZ = 1200.0
_F0_JUMP_WARN_HZ = 35.0
_F0_JUMP_HARD_HZ = 70.0
# Tight overlap or near-zero gap = click risk. Leave ≥20 ms of headroom.
_GAP_TIGHT_WARN_MS = 20
_GAP_TIGHT_HARD_MS = 0   # overlap
# Dead-air within a phrase-group (phrases <=300 ms apart conceptually one idea)
_GAP_DEAD_WARN_MS = 400
_GAP_DEAD_HARD_MS = 800

# Analysis window around each boundary (±this many ms). The windows are
# **anchored to the actual phrase audio**, not to the boundary midpoint —
# that way the pre-window measures the tail of the previous phrase and
# the post-window measures the head of the next one, without the
# inter-phrase silence/fade contaminating either.
#
# 400 ms (antes 250, antes 120): una ventana corta mide fonemas
# puntuales (vocal tónica vs consonante sorda) que dan saltos de 15+
# dB aun con el RMS global nivelado. 400 ms promedia un bloque
# prosódico humano completo (sílaba tónica + coda, ~300-400 ms) →
# mide la energía PERCIBIDA, que es lo que el oyente interpreta
# como "salto" entre frases. A 250 ms todavía pegaba un fonema de
# ataque fuerte contra una cola débil de la frase anterior.
_WINDOW_MS = 400


@dataclass
class BoundaryIssue:
    """One boundary between TTS segments with its measured discontinuities."""

    index: int                     # boundary index (0 = between seg 0 and 1)
    timestamp_ms: int              # where the boundary sits on the timeline
    gap_ms: int                    # start_{n+1} - end_n (can be negative)
    rms_jump_db: Optional[float]
    centroid_jump_hz: Optional[float]
    f0_jump_hz: Optional[float]
    severity: str                  # "ok" | "warn" | "hard"
    reasons: list[str] = field(default_factory=list)


@dataclass
class BoundaryReport:
    """Aggregate report over all boundaries on a timeline."""

    total_boundaries: int
    hard_cuts: int
    warnings: int
    worst_boundary_idx: Optional[int]
    issues: list[BoundaryIssue]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def analyze_boundaries(
    samples: np.ndarray,
    sample_rate: int,
    boundaries: list[tuple[int, int]],
) -> Optional[BoundaryReport]:
    """Analyze every boundary on a mono float32 audio buffer.

    Parameters
    ----------
    samples :
        1-D float array, mono, normalised to [-1, 1]. Stereo should be
        mixed down by the caller.
    sample_rate :
        Sample rate of ``samples``.
    boundaries :
        List of ``(end_ms_of_prev, start_ms_of_next)`` for every adjacent
        pair of TTS segments on the final timeline (after overlap
        resolution and inter-phrase fades).

    Returns
    -------
    BoundaryReport or None
        ``None`` iff a hard dependency is missing (librosa) — the pipeline
        then records a blank QA entry instead of failing.
    """
    try:
        import librosa  # noqa: F401
    except ImportError:
        logger.warning("librosa unavailable; skipping boundary QA")
        return None

    if not boundaries:
        return BoundaryReport(
            total_boundaries=0, hard_cuts=0, warnings=0,
            worst_boundary_idx=None, issues=[],
        )

    issues: list[BoundaryIssue] = []
    hard = warn = 0
    worst_score = -1.0
    worst_idx: Optional[int] = None

    for i, (end_prev_ms, start_next_ms) in enumerate(boundaries):
        gap_ms = start_next_ms - end_prev_ms
        # Timestamp reported to the frontend = the audible boundary point.
        # Mid-point reads most naturally on the timeline overlay.
        boundary_mid_ms = (end_prev_ms + start_next_ms) // 2

        # Anchor the analysis windows to the *phrase audio*, not to the
        # gap. ``pre`` = last _WINDOW_MS of the previous phrase; ``post``
        # = first _WINDOW_MS of the next phrase. Skipping the gap avoids
        # measuring silence/fade against voice — that false-positive was
        # responsible for most "salto RMS" / "salto timbre" reports.
        pre_slice = _slice_ms(
            samples, sample_rate,
            end_prev_ms - _WINDOW_MS, end_prev_ms,
        )
        post_slice = _slice_ms(
            samples, sample_rate,
            start_next_ms, start_next_ms + _WINDOW_MS,
        )

        rms_jump = _rms_jump_db(pre_slice, post_slice)
        centroid_jump = _centroid_jump_hz(pre_slice, post_slice, sample_rate)
        f0_jump = _f0_jump_hz(pre_slice, post_slice, sample_rate)

        severity, reasons, score = _classify(
            gap_ms, rms_jump, centroid_jump, f0_jump,
        )

        issues.append(BoundaryIssue(
            index=i,
            timestamp_ms=boundary_mid_ms,
            gap_ms=gap_ms,
            rms_jump_db=_round(rms_jump, 2),
            centroid_jump_hz=_round(centroid_jump, 1),
            f0_jump_hz=_round(f0_jump, 1),
            severity=severity,
            reasons=reasons,
        ))
        if severity == "hard":
            hard += 1
        elif severity == "warn":
            warn += 1
        if score > worst_score:
            worst_score = score
            worst_idx = i

    return BoundaryReport(
        total_boundaries=len(issues),
        hard_cuts=hard,
        warnings=warn,
        worst_boundary_idx=worst_idx,
        issues=issues,
    )


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

def _slice_ms(
    samples: np.ndarray, sr: int, start_ms: int, end_ms: int,
) -> np.ndarray:
    start = max(0, int(start_ms * sr / 1000))
    end = min(len(samples), int(end_ms * sr / 1000))
    if end <= start:
        return np.zeros(1, dtype=samples.dtype)
    return samples[start:end]


def _rms_jump_db(pre: np.ndarray, post: np.ndarray) -> Optional[float]:
    if pre.size < 2 or post.size < 2:
        return None
    rms_pre = float(np.sqrt(np.mean(pre.astype(np.float64) ** 2)))
    rms_post = float(np.sqrt(np.mean(post.astype(np.float64) ** 2)))
    if rms_pre <= 1e-6 or rms_post <= 1e-6:
        # Either side is near-silent — not a "jump", just a fade into/out
        # of pause. Those are handled by the gap metric.
        return None
    return abs(20.0 * math.log10(rms_post / rms_pre))


def _centroid_jump_hz(
    pre: np.ndarray, post: np.ndarray, sr: int,
) -> Optional[float]:
    import librosa
    if pre.size < 512 or post.size < 512:
        return None
    try:
        c_pre = float(np.mean(
            librosa.feature.spectral_centroid(y=pre, sr=sr)[0]
        ))
        c_post = float(np.mean(
            librosa.feature.spectral_centroid(y=post, sr=sr)[0]
        ))
    except Exception:
        return None
    return abs(c_post - c_pre)


def _f0_jump_hz(
    pre: np.ndarray, post: np.ndarray, sr: int,
) -> Optional[float]:
    """Mean F0 difference using librosa.pyin (voiced frames only)."""
    import librosa
    if pre.size < sr // 10 or post.size < sr // 10:
        return None
    try:
        f0_pre, vp, _ = librosa.pyin(
            pre, fmin=70.0, fmax=400.0, sr=sr, frame_length=1024,
        )
        f0_post, vq, _ = librosa.pyin(
            post, fmin=70.0, fmax=400.0, sr=sr, frame_length=1024,
        )
    except Exception:
        return None

    pre_v = f0_pre[np.isfinite(f0_pre)] if f0_pre is not None else None
    post_v = f0_post[np.isfinite(f0_post)] if f0_post is not None else None
    if pre_v is None or post_v is None or pre_v.size < 3 or post_v.size < 3:
        return None
    return float(abs(np.median(pre_v) - np.median(post_v)))


def _classify(
    gap_ms: int,
    rms_jump: Optional[float],
    centroid_jump: Optional[float],
    f0_jump: Optional[float],
) -> tuple[str, list[str], float]:
    """Map metrics to severity + human-readable reasons + numeric score.

    Score is only used to pick the *worst* boundary for the report; it is
    not exposed. Larger = worse.
    """
    reasons: list[str] = []
    severity = "ok"
    score = 0.0

    def bump(level: str, reason: str, s: float) -> None:
        nonlocal severity, score
        reasons.append(reason)
        score += s
        if level == "hard" or severity == "hard":
            severity = "hard"
        elif level == "warn" and severity != "hard":
            severity = "warn"

    # Gap — both too tight and too loose are issues.
    if gap_ms <= _GAP_TIGHT_HARD_MS:
        bump("hard", f"overlap ({gap_ms} ms)", 3.0)
    elif gap_ms < _GAP_TIGHT_WARN_MS:
        bump("warn", f"gap muy ajustado ({gap_ms} ms)", 1.0)
    if gap_ms >= _GAP_DEAD_HARD_MS:
        bump("hard", f"silencio largo ({gap_ms} ms)", 2.5)
    elif gap_ms >= _GAP_DEAD_WARN_MS:
        bump("warn", f"silencio perceptible ({gap_ms} ms)", 1.0)

    # RMS jump — loudness step across the boundary.
    if rms_jump is not None:
        if rms_jump >= _RMS_JUMP_HARD_DB:
            bump("hard", f"salto RMS {rms_jump:.1f} dB", rms_jump)
        elif rms_jump >= _RMS_JUMP_WARN_DB:
            bump("warn", f"salto RMS {rms_jump:.1f} dB", rms_jump / 2)

    # Spectral centroid — timbre / brightness jump.
    if centroid_jump is not None:
        if centroid_jump >= _CENTROID_JUMP_HARD_HZ:
            bump("hard", f"salto timbre {centroid_jump:.0f} Hz", centroid_jump / 200)
        elif centroid_jump >= _CENTROID_JUMP_WARN_HZ:
            bump("warn", f"salto timbre {centroid_jump:.0f} Hz", centroid_jump / 400)

    # F0 — pitch jump. The clearest "otra voz" signal.
    if f0_jump is not None:
        if f0_jump >= _F0_JUMP_HARD_HZ:
            bump("hard", f"salto pitch {f0_jump:.0f} Hz", f0_jump / 20)
        elif f0_jump >= _F0_JUMP_WARN_HZ:
            bump("warn", f"salto pitch {f0_jump:.0f} Hz", f0_jump / 40)

    return severity, reasons, score


def _round(v: Optional[float], ndigits: int) -> Optional[float]:
    return round(v, ndigits) if v is not None else None
