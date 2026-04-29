"""Automatic QA for dubbed audio output.

Two complementary layers:

* :mod:`boundary_report` — objective per-boundary metrics (gap, RMS jump,
  spectral centroid jump, F0 jump). Runs in milliseconds, no models; flags
  where a listener will hear a hard cut between phrases.
* :mod:`mos_scorer` — UTMOS neural MOS predictor over the whole wav. One
  number 1-5 that correlates with perceived naturalness. Covers the
  "voice sounds robotic" axis that boundary metrics can't see.

Both layers are best-effort: if a dependency is missing or inference
fails, they return ``None`` so the dubbing pipeline never fails because
of QA. Results are persisted to ``{base}.dub-qa.json`` next to the video.
"""

from .boundary_report import BoundaryReport, analyze_boundaries
from .mos_scorer import MosScore, score_mos

__all__ = [
    "BoundaryReport",
    "MosScore",
    "analyze_boundaries",
    "score_mos",
]
