"""Unit tests for SynthesizerPiper — mocks the PiperVoice class."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dubbing_generator.config import DubbingConfig


class _FakeChunk:
    """Mimics piper.voice.AudioChunk for tests."""

    def __init__(self, n_samples: int, sample_rate: int = 22050):
        self.audio_int16_bytes = struct.pack("<" + "h" * n_samples, *([0] * n_samples))
        self.sample_rate = sample_rate
        self.sample_width = 2
        self.sample_channels = 1


@pytest.fixture
def cfg(tmp_path):
    fake_model = tmp_path / "voice.onnx"
    fake_model.write_bytes(b"fake onnx")
    return DubbingConfig(tts_engine="piper", piper_model_path=str(fake_model))


@pytest.fixture
def ref_wav(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(b"RIFFxxxxWAVEfmt ")
    return p


def _install_fake_voice(synth, n_samples=4410, sample_rate=22050):
    fake_voice = MagicMock()
    fake_voice.synthesize.return_value = iter([_FakeChunk(n_samples, sample_rate)])
    synth._voice = fake_voice
    return fake_voice


def test_generate_empty_text_returns_silence(cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_piper import SynthesizerPiper
    synth = SynthesizerPiper(cfg)
    out = synth.generate("   ", ref_wav)
    assert len(out) == 100


def test_generate_returns_audiosegment_at_24khz(cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_piper import SynthesizerPiper
    synth = SynthesizerPiper(cfg)
    _install_fake_voice(synth, n_samples=4410, sample_rate=22050)  # ~200 ms

    out = synth.generate("hola mundo", ref_wav)

    assert out.frame_rate == 24000
    assert out.channels == 1
    assert out.sample_width == 2
    assert 150 <= len(out) <= 250, f"expected ~200 ms, got {len(out)} ms"


def test_generate_passes_length_scale_from_speed(cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_piper import SynthesizerPiper
    synth = SynthesizerPiper(cfg)
    fake = _install_fake_voice(synth)

    synth.generate("texto prueba", ref_wav, speed=2.0)

    args, kwargs = fake.synthesize.call_args
    syn_config = kwargs.get("syn_config") or (args[1] if len(args) > 1 else None)
    assert syn_config is not None
    assert syn_config.length_scale == pytest.approx(0.5)
    assert syn_config.noise_scale == cfg.piper_noise_scale
    assert syn_config.noise_w_scale == cfg.piper_noise_w


def test_generate_falls_back_on_synth_error(cfg, ref_wav, caplog):
    from dubbing_generator.tts.synthesizer_piper import SynthesizerPiper
    synth = SynthesizerPiper(cfg)
    fake_voice = MagicMock()
    fake_voice.synthesize.side_effect = RuntimeError("onnx fail")
    synth._voice = fake_voice

    out = synth.generate("any text", ref_wav)

    assert len(out) == 200
    assert any("Piper synthesize failed" in r.message for r in caplog.records)


def test_generate_empty_chunks_returns_silence(cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_piper import SynthesizerPiper
    synth = SynthesizerPiper(cfg)
    fake_voice = MagicMock()
    fake_voice.synthesize.return_value = iter([])
    synth._voice = fake_voice

    out = synth.generate("anything", ref_wav)

    assert len(out) == 200


def test_missing_model_raises(tmp_path, ref_wav):
    from dubbing_generator.tts.synthesizer_piper import SynthesizerPiper
    cfg = DubbingConfig(
        tts_engine="piper",
        piper_model_path=str(tmp_path / "missing.onnx"),
    )
    synth = SynthesizerPiper(cfg)
    with pytest.raises(RuntimeError, match="Piper model not found"):
        synth._get_voice()


def test_build_synthesizer_dispatches_piper(tmp_path):
    from dubbing_generator.tts import build_synthesizer
    from dubbing_generator.tts.synthesizer_piper import SynthesizerPiper
    fake_model = tmp_path / "voice.onnx"
    fake_model.write_bytes(b"x")
    cfg = DubbingConfig(tts_engine="piper", piper_model_path=str(fake_model))
    synth = build_synthesizer(cfg)
    assert isinstance(synth, SynthesizerPiper)
