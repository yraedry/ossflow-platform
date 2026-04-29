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
