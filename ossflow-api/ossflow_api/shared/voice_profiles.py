"""Resolución del perfil de voz a partir del sidecar ``.bjj-meta.json``.

Antes esta función vivía como ``_load_voice_profile_for_path`` dentro de
``api/pipeline.py``, e ``api/dubbing.py`` la importaba como símbolo
privado — un acoplamiento sucio entre módulos. Al moverla a ``shared/``
ambos módulos importan una API pública sin atravesar fronteras de feature.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

SIDECAR_NAME = ".bjj-meta.json"
"""Nombre del sidecar de metadatos por instructional."""

_MAX_LEVELS_UP = 4
"""Niveles máximos para subir buscando el sidecar (Season / chapters / file)."""


def load_voice_profile_for_path(host_path: str) -> str:
    """Sube niveles desde el path del vídeo buscando ``voice_profile`` en el sidecar.

    Los vídeos viven a varios niveles de profundidad (``Season_NN/`` con los
    capítulos), así que el sidecar suele estar 1-2 niveles arriba. Cadena
    vacía significa "clonar al instructor" (comportamiento por defecto).
    """
    p = Path(host_path)
    current = p if p.is_dir() else p.parent
    for _ in range(_MAX_LEVELS_UP):
        sidecar = current / SIDECAR_NAME
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    vp = data.get("voice_profile")
                    if isinstance(vp, str) and vp:
                        return vp
            except (OSError, ValueError) as exc:
                log.warning("Failed to read voice_profile from %s: %s", sidecar, exc)
        if current.parent == current:
            break
        current = current.parent
    return ""
