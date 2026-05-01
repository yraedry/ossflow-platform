"""QA helpers del pipeline de dubbing.

Migrado de ``pipeline.py`` en T32.3. Por ahora contiene solo
``compute_verdict``; el resto de helpers QA (``_run_qa``,
``_compute_boundaries``, etc.) viven aún en ``DubbingPipeline``
porque dependen del state de la instancia.
"""

from __future__ import annotations

from typing import Any


def compute_verdict(boundary_report, mos) -> dict[str, Any]:
    """Combina señales de boundary + MOS en un veredicto green/amber/red.

    Boundary-first: los boundaries reflejan cortes que el listener
    percibe; el UTMOS infravalora speech doblado-ducked (entrenado en
    audio limpio). MOS es desempate, no gate.

    * **red**   — al menos un hard boundary (corte real audible).
    * **amber** — solo warnings (potencialmente audible, no jarring).
    * **green** — cero boundaries flagged.

    MOS solo eleva severidad si está muy bajo (< 2.0) y ya éramos
    amber — un capítulo con cero boundaries pero MOS 2.1 (normal
    para TTS+ducking) se mantiene green.
    """
    hard = boundary_report.hard_cuts if boundary_report else 0
    warn = boundary_report.warnings if boundary_report else 0
    mos_score = mos.score if mos else None

    if hard > 0:
        level = "red"
    elif warn > 0:
        level = "amber"
    else:
        level = "green"

    if mos_score is not None and mos_score < 2.0 and level == "amber":
        level = "red"

    return {
        "level": level,
        "mos": mos_score,
        "hard_cuts": hard,
        "warnings": warn,
    }
