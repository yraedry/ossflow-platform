"""DTOs y constantes del módulo metadata."""

from __future__ import annotations

from typing import Any

# Forma canónica del sidecar ``.bjj-meta.json``. La incluimos como dict para
# poder copiarla con ``copy.deepcopy`` y proteger ``tags`` contra mutaciones
# del cliente. Idéntica al pre-refactor.
DEFAULT_METADATA: dict[str, Any] = {
    "instructor": "",
    "topic": "",
    "tags": [],
    "synopsis": "",
    "year": None,
    # voice_profile: filename bajo /voices (ej. "narrador_es.wav") o "" para
    # clonar la voz del propio instructor. Lo lee pipeline.py al encolar el
    # paso de doblaje para este instructional.
    "voice_profile": "",
}
