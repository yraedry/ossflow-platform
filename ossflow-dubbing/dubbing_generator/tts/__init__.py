"""TTS subpackage: voice synthesis backends.

Tras T22.5 (limpieza arquitectónica del 2026-04-30), el sistema soporta
**un único motor: S2-Pro** (Fish Audio S2-Pro, voice cloning local con
backend Vulkan). Los motores ElevenLabs/Piper/Kokoro fueron eliminados
porque eran pruebas y no se mantenían.

El factory ``build_synthesizer`` mantiene la forma extensible (``if
engine != 's2pro': raise``) para que añadir un motor 2 en el futuro
cueste 5 LOC, sin necesidad de mantener los motores eliminados.
"""

from __future__ import annotations

from ..config import DubbingConfig


def build_synthesizer(cfg: DubbingConfig, server_manager=None):
    """Devuelve la instancia del sintetizador configurado.

    ``server_manager`` se pasa al motor para que pueda resucitar el
    subproceso tras un crash mid-job. Solo S2-Pro lo usa actualmente.
    """
    engine = cfg.tts_engine
    if engine != "s2pro":
        raise ValueError(
            f"Unsupported tts_engine: {engine!r}. "
            f"Solo 's2pro' está soportado tras T22.5."
        )
    from .synthesizer_s2pro import SynthesizerS2Pro
    return SynthesizerS2Pro(cfg, server_manager=server_manager)
