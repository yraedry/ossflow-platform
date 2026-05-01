"""Tests del factory ``build_synthesizer``.

Tras T22.5: factory mono-motor. Cualquier motor distinto de ``s2pro``
lanza ``ValueError`` con el mensaje preservado para que sea evidente la
política del sistema.
"""

import pytest

from dubbing_generator.config import DubbingConfig
from dubbing_generator.tts import build_synthesizer
from dubbing_generator.tts.synthesizer_s2pro import SynthesizerS2Pro


def test_factory_returns_s2pro_for_s2pro_engine():
    cfg = DubbingConfig(tts_engine="s2pro")
    inst = build_synthesizer(cfg)
    try:
        assert isinstance(inst, SynthesizerS2Pro)
    finally:
        inst.close()


def test_factory_rejects_unsupported_engine():
    """ElevenLabs / Piper / Kokoro / cualquier otro → ValueError."""
    for engine in ("elevenlabs", "piper", "kokoro", "xtts", "unknown"):
        cfg = DubbingConfig(tts_engine=engine)
        with pytest.raises(ValueError, match="Unsupported tts_engine"):
            build_synthesizer(cfg)
