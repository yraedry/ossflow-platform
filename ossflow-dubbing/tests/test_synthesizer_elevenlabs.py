"""Unit tests for SynthesizerElevenLabs — mocks the ElevenLabs SDK."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dubbing_generator.config import DubbingConfig


def _pcm_bytes(n_samples: int = 2400) -> bytes:
    """Generate n_samples of silence as pcm_s16le (2400 = 100 ms @ 24 kHz)."""
    return struct.pack("<" + "h" * n_samples, *([0] * n_samples))


@pytest.fixture
def cfg():
    return DubbingConfig(tts_engine="elevenlabs")


@pytest.fixture
def ref_wav(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(b"RIFFxxxxWAVEfmt ")
    return p


def _install_fake_client(synth, raw_audio: bytes | list[bytes]):
    """Replace the lazy client with a mock that returns the given bytes."""
    fake_client = MagicMock()
    if isinstance(raw_audio, list):
        fake_client.text_to_speech.convert.return_value = iter(raw_audio)
    else:
        fake_client.text_to_speech.convert.return_value = iter([raw_audio])
    synth._client = fake_client
    # Bypass VoiceSettings import in tests too
    synth._voice_settings = MagicMock()
    return fake_client


def test_generate_empty_text_returns_silence(cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_elevenlabs import SynthesizerElevenLabs
    synth = SynthesizerElevenLabs(cfg)
    out = synth.generate("   ", ref_wav)
    assert len(out) == 100


def test_generate_pcm_24000_returns_audiosegment(cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_elevenlabs import SynthesizerElevenLabs
    synth = SynthesizerElevenLabs(cfg)
    fake = _install_fake_client(synth, _pcm_bytes(2400))  # 100 ms

    out = synth.generate("hola mundo", ref_wav)

    assert 80 <= len(out) <= 120, f"expected ~100 ms, got {len(out)} ms"
    assert out.frame_rate == 24000
    assert out.channels == 1
    fake.text_to_speech.convert.assert_called_once()
    call_kwargs = fake.text_to_speech.convert.call_args.kwargs
    assert call_kwargs["voice_id"] == cfg.elevenlabs_voice_id
    assert call_kwargs["model_id"] == cfg.elevenlabs_model_id
    assert call_kwargs["output_format"] == "pcm_24000"
    assert call_kwargs["text"] == "hola mundo"


def test_generate_handles_chunk_iterator(cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_elevenlabs import SynthesizerElevenLabs
    synth = SynthesizerElevenLabs(cfg)
    chunks = [_pcm_bytes(1200), _pcm_bytes(1200)]  # 50 ms + 50 ms
    _install_fake_client(synth, chunks)

    out = synth.generate("dos chunks", ref_wav)

    assert 80 <= len(out) <= 120


def test_generate_falls_back_on_api_error(cfg, ref_wav, caplog):
    from dubbing_generator.tts.synthesizer_elevenlabs import SynthesizerElevenLabs
    synth = SynthesizerElevenLabs(cfg)
    fake_client = MagicMock()
    fake_client.text_to_speech.convert.side_effect = RuntimeError("network down")
    synth._client = fake_client
    synth._voice_settings = MagicMock()

    out = synth.generate("any text", ref_wav)

    assert len(out) == 200
    assert any("ElevenLabs convert failed" in r.message for r in caplog.records)


def test_generate_empty_bytes_returns_silence(cfg, ref_wav, caplog):
    from dubbing_generator.tts.synthesizer_elevenlabs import SynthesizerElevenLabs
    synth = SynthesizerElevenLabs(cfg)
    _install_fake_client(synth, b"")

    out = synth.generate("something", ref_wav)

    assert len(out) == 200
    assert any("empty audio" in r.message for r in caplog.records)


def test_missing_api_key_raises(monkeypatch, cfg, ref_wav):
    from dubbing_generator.tts.synthesizer_elevenlabs import SynthesizerElevenLabs
    monkeypatch.delenv(cfg.elevenlabs_api_key_env, raising=False)
    synth = SynthesizerElevenLabs(cfg)

    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        synth._get_client()


def test_build_synthesizer_dispatches_elevenlabs():
    from dubbing_generator.tts import build_synthesizer
    from dubbing_generator.tts.synthesizer_elevenlabs import SynthesizerElevenLabs
    cfg = DubbingConfig(tts_engine="elevenlabs")
    synth = build_synthesizer(cfg)
    assert isinstance(synth, SynthesizerElevenLabs)


def test_build_synthesizer_unknown_engine_raises():
    from dubbing_generator.tts import build_synthesizer
    cfg = DubbingConfig(tts_engine="bogus")
    with pytest.raises(ValueError, match="Unsupported tts_engine"):
        build_synthesizer(cfg)
