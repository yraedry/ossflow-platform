"""Unit tests for SynthesizerKokoro — mocks the KPipeline class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dubbing_generator.config import DubbingConfig


@pytest.fixture
def cfg():
    return DubbingConfig(tts_engine="kokoro")


@pytest.fixture
def ref_wav(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(b"RIFFxxxxWAVEfmt ")
    return p


def _fake_pipeline_output(n_samples=4800):
    """Return iterable of (gs, ps, audio) tuples mimicking KPipeline."""
    audio = np.zeros(n_samples, dtype=np.float32)
    return [(0, "phon", audio)]


def _install_fake_pipeline(synth, n_samples=4800):
    fake = MagicMock()
    fake.return_value = _fake_pipeline_output(n_samples)
    synth._pipeline = fake
    return fake


def test_generate_empty_text_returns_silence(cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_kokoro import SynthesizerKokoro
    synth = SynthesizerKokoro(cfg)
    out = synth.generate("   ", ref_wav)
    assert len(out) == 100


def test_generate_returns_audiosegment_at_24khz(cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_kokoro import SynthesizerKokoro
    synth = SynthesizerKokoro(cfg)
    _install_fake_pipeline(synth, n_samples=4800)  # 200 ms @ 24kHz

    out = synth.generate("hola mundo", ref_wav)

    assert out.frame_rate == 24000
    assert out.channels == 1
    assert out.sample_width == 2
    assert 150 <= len(out) <= 250


def test_generate_uses_voice_and_speed_from_config(cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_kokoro import SynthesizerKokoro
    synth = SynthesizerKokoro(cfg)
    fake = _install_fake_pipeline(synth)

    synth.generate("texto prueba", ref_wav)

    args, kwargs = fake.call_args
    assert kwargs["voice"] == cfg.kokoro_voice
    assert kwargs["speed"] == cfg.kokoro_speed


def test_generate_speed_override(cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_kokoro import SynthesizerKokoro
    synth = SynthesizerKokoro(cfg)
    fake = _install_fake_pipeline(synth)

    synth.generate("texto", ref_wav, speed=1.5)

    kwargs = fake.call_args.kwargs
    assert kwargs["speed"] == 1.5


def test_generate_falls_back_on_synth_error(cfg, ref_wav, caplog):
    from dubbing_generator.tts.synthesizer_kokoro import SynthesizerKokoro
    synth = SynthesizerKokoro(cfg)
    fake = MagicMock()
    fake.side_effect = RuntimeError("kokoro fail")
    synth._pipeline = fake

    out = synth.generate("any text", ref_wav)

    assert len(out) == 200
    assert any("Kokoro synthesize failed" in r.message for r in caplog.records)


def test_build_synthesizer_dispatches_kokoro():
    from dubbing_generator.tts import build_synthesizer
    from dubbing_generator.tts.synthesizer_kokoro import SynthesizerKokoro
    cfg = DubbingConfig(tts_engine="kokoro")
    synth = build_synthesizer(cfg)
    assert isinstance(synth, SynthesizerKokoro)
