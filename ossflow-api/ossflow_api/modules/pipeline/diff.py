"""Snapshot + diff de directorios.

Migrado de ``api/pipeline.py`` en T_LATE_2.3. Funciones puras — el
único acoplamiento al runner es ``target_dir`` que recibe un
``PipelineInfo``.

* ``snapshot_dir(base)`` mapea cada fichero bajo ``base`` a
  ``(size, mtime)``. Diff = comparar dos snapshots.
* ``compute_diff(before, after)`` clasifica added/modified/removed
  con truncado opcional.
* ``detect_season_folder(target, added)`` heurística: dada una lista
  de paths relativos añadidos, devuelve la carpeta Season más
  probable.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional

from .schemas import PipelineInfo


VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")
SEASON_DIR_RE = re.compile(r"^Season\s*\d+$", re.IGNORECASE)


def target_dir(pipeline: PipelineInfo) -> Optional[Path]:
    """Directorio host a snapshotear (output_dir o el dir del vídeo)."""
    out = pipeline.options.get("output_dir")
    if out:
        p = Path(out)
    else:
        pp = Path(pipeline.path)
        p = pp if pp.is_dir() else pp.parent
    try:
        if p.exists() and p.is_dir():
            return p
    except OSError:
        return None
    return None


def snapshot_dir(base: Optional[Path]) -> dict[str, tuple[int, float]]:
    """Devuelve ``{relative_path: (size, mtime)}`` para todos los ficheros."""
    if base is None:
        return {}
    out: dict[str, tuple[int, float]] = {}
    try:
        for f in base.rglob("*"):
            try:
                if not f.is_file():
                    continue
                st = f.stat()
                rel = f.relative_to(base).as_posix()
                out[rel] = (st.st_size, st.st_mtime)
            except OSError:
                continue
    except OSError:
        return {}
    return out


def compute_diff(
    before: dict[str, tuple[int, float]],
    after: dict[str, tuple[int, float]],
    limit: int = 200,
) -> dict:
    added_all = [p for p in after if p not in before]
    removed_all = [p for p in before if p not in after]
    modified_all = [
        p for p in after
        if p in before
        and (
            before[p][0] != after[p][0] or abs(before[p][1] - after[p][1]) > 0.001
        )
    ]

    def _trunc(lst):
        return lst[:limit], len(lst) > limit

    added, a_t = _trunc(sorted(added_all))
    modified, m_t = _trunc(sorted(modified_all))
    removed, r_t = _trunc(sorted(removed_all))
    return {
        "added": added,
        "modified": modified,
        "removed": removed,
        "truncated": a_t or m_t or r_t,
    }


def detect_season_folder(
    target: Optional[Path],
    added: list[str],
) -> Optional[str]:
    """A partir de la lista ``added``, devuelve el path absoluto de la
    Season folder donde aterrizaron capítulos nuevos.

    Heurística: agrupa por parent dir; prefiere parents que matchean
    ``Season NN``; desempata por count.
    """
    if target is None or not added:
        return None
    parents: Counter = Counter()
    for rel in added:
        if not rel.lower().endswith(VIDEO_EXTS):
            continue
        parts = rel.rsplit("/", 1)
        if len(parts) != 2:
            continue
        parents[parts[0]] += 1
    if not parents:
        return None
    season_like = {
        p: c for p, c in parents.items() if SEASON_DIR_RE.match(Path(p).name)
    }
    chosen = max(season_like or parents, key=lambda k: (season_like or parents)[k])
    return str(target / chosen)
