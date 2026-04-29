"""TTS subpackage: voice synthesis backends."""

from __future__ import annotations

from ..config import DubbingConfig


def build_synthesizer(cfg: DubbingConfig, server_manager=None):
    """Return the configured synthesizer instance.

    Supported engines:
    - ``s2pro`` (local Vulkan voice cloning, default since 2026-04-27)
    - ``elevenlabs`` (cloud, voice cloning, paid)
    - ``piper`` (local ONNX, no cloning, free, fast)
    - ``kokoro`` (local StyleTTS2, no cloning, free, GPU)

    ``server_manager`` is passed through to engines that own a subprocess
    so they can resurrect it after a crash mid-job. Currently only s2pro
    uses it; other engines accept the kwarg as a no-op.
    """
    engine = cfg.tts_engine
    if engine == "s2pro":
        from .synthesizer_s2pro import SynthesizerS2Pro
        return SynthesizerS2Pro(cfg, server_manager=server_manager)
    if engine == "elevenlabs":
        from .synthesizer_elevenlabs import SynthesizerElevenLabs
        return SynthesizerElevenLabs(cfg)
    if engine == "piper":
        from .synthesizer_piper import SynthesizerPiper
        return SynthesizerPiper(cfg)
    if engine == "kokoro":
        from .synthesizer_kokoro import SynthesizerKokoro
        return SynthesizerKokoro(cfg)
    raise ValueError(
        f"Unsupported tts_engine: {engine!r} "
        f"(supported: 's2pro', 'elevenlabs', 'piper', 'kokoro')"
    )
