from __future__ import annotations

import io
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydub import AudioSegment

from dubbing_generator.config import DubbingConfig
from dubbing_generator.tts.synthesizer_s2pro import SynthesizerS2Pro


def _wav_bytes(duration_ms: int = 600, freq: int = 440) -> bytes:
    sr = 44100
    n = sr * duration_ms // 1000
    t = np.arange(n) / sr
    pcm = (np.sin(2 * np.pi * freq * t) * 32000).astype(np.int16).tobytes()
    seg = AudioSegment(pcm, frame_rate=sr, sample_width=2, channels=1)
    buf = io.BytesIO(); seg.export(buf, format="wav")
    return buf.getvalue()


def test_returns_silence_for_empty_text(tmp_path: Path):
    cfg = DubbingConfig(tts_engine="s2pro",
                        s2_ref_audio_path=str(tmp_path / "missing.wav"))
    s = SynthesizerS2Pro(cfg)
    out = s.generate("   ", reference_wav=tmp_path / "missing.wav")
    assert isinstance(out, AudioSegment)
    assert len(out) == 100


def test_returns_silence_when_ref_wav_missing(tmp_path: Path):
    cfg = DubbingConfig(tts_engine="s2pro",
                        s2_ref_audio_path=str(tmp_path / "missing.wav"))
    s = SynthesizerS2Pro(cfg)
    out = s.generate("hola mundo", reference_wav=tmp_path / "missing.wav")
    assert len(out) == 200


def test_posts_to_server_with_correct_form(tmp_path: Path):
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"\0" * 16)
    cfg = DubbingConfig(
        tts_engine="s2pro",
        s2_ref_audio_path=str(ref),
        s2_server_host="127.0.0.1",
        s2_server_port=3030,
    )
    fake_wav = _wav_bytes(duration_ms=600)
    fake_resp = MagicMock(); fake_resp.status_code = 200
    fake_resp.content = fake_wav; fake_resp.raise_for_status = lambda: None

    s = SynthesizerS2Pro(cfg)
    with patch.object(s._client, "post", return_value=fake_resp) as post:
        out = s.generate("hola mundo", reference_wav=ref)

    # Accept either positional or keyword url for robustness.
    call_args = post.call_args
    url = call_args.args[0] if call_args.args else call_args.kwargs["url"]
    assert url == "/generate"
    files = call_args.kwargs["files"]
    data = call_args.kwargs["data"]
    assert "reference" in files
    assert data["text"] == "hola mundo"
    assert data["reference_text"] == cfg.s2_ref_text
    params = json.loads(data["params"])
    assert params["temperature"] == 0.8
    assert params["top_p"] == 0.8
    assert params["top_k"] == 30
    assert isinstance(out, AudioSegment)
    assert 550 <= len(out) <= 650


def test_returns_silence_on_http_error(tmp_path: Path):
    import httpx
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"\0" * 16)
    cfg = DubbingConfig(tts_engine="s2pro", s2_ref_audio_path=str(ref))
    s = SynthesizerS2Pro(cfg)
    with patch.object(s._client, "post",
                      side_effect=httpx.ConnectError("server down")):
        out = s.generate("hola", reference_wav=ref)
    assert isinstance(out, AudioSegment)
    assert len(out) == 200


def test_close_closes_client():
    cfg = DubbingConfig(tts_engine="s2pro")
    s = SynthesizerS2Pro(cfg)
    with patch.object(s._client, "close") as close:
        s.close()
        close.assert_called_once()


def test_resurrects_server_after_connection_refused(tmp_path: Path):
    """When the s2.cpp subprocess dies mid-job (Connection refused / Server
    disconnected), the synthesizer should ask its manager to restart the
    server before the next phrase, rather than burning the whole 60-second
    breaker cooldown emitting silence."""
    import httpx
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"\0" * 16)
    cfg = DubbingConfig(tts_engine="s2pro", s2_ref_audio_path=str(ref))

    manager = MagicMock()
    manager.wait_until_ready.return_value = True
    s = SynthesizerS2Pro(cfg, server_manager=manager)

    fake_wav = _wav_bytes(duration_ms=400)
    fake_resp = MagicMock(); fake_resp.status_code = 200
    fake_resp.content = fake_wav; fake_resp.raise_for_status = lambda: None

    # First call: server crashed mid-response. Synth records _needs_restart.
    # Second call: synth notices the flag, restarts via manager, then succeeds.
    side_effects = [
        httpx.RemoteProtocolError("Server disconnected without sending a response"),
        fake_resp,
    ]
    with patch.object(s._client, "post", side_effect=side_effects):
        first = s.generate("frase larga que crashea", reference_wav=ref)
        second = s.generate("frase siguiente que ya va bien", reference_wav=ref)

    assert len(first) == 200, "first call returns silence on disconnect"
    assert manager.stop.called, "manager.stop must be called before restart"
    assert manager.start.called, "manager.start must be called to revive"
    assert isinstance(second, AudioSegment)
    assert 350 <= len(second) <= 450, "second call produces real audio"


def test_circuit_breaker_opens_after_consecutive_failures(tmp_path: Path):
    """After 3 consecutive HTTP errors, the synthesizer fails fast for 1
    minute rather than hammering the server."""
    import httpx
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"\0" * 16)
    cfg = DubbingConfig(tts_engine="s2pro", s2_ref_audio_path=str(ref))
    s = SynthesizerS2Pro(cfg)
    with patch.object(s._client, "post",
                      side_effect=httpx.ConnectError("server down")) as post:
        for _ in range(3):
            s.generate("hola", reference_wav=ref)
        assert post.call_count == 3
        # 4th call must short-circuit
        out = s.generate("hola", reference_wav=ref)
        assert post.call_count == 3
        assert len(out) == 200


@pytest.mark.skipif(
    os.environ.get("S2_SMOKE") != "1",
    reason="requires running s2.cpp server (set S2_SMOKE=1)",
)
def test_smoke_against_running_server(tmp_path: Path):
    cfg = DubbingConfig(tts_engine="s2pro")
    ref = Path(cfg.s2_ref_audio_path)
    if not ref.exists():
        pytest.skip(f"ref voice not available at {ref}")
    s = SynthesizerS2Pro(cfg)
    out = s.generate(
        "Cuando atrapas el cuello, tu codo va por encima del hombro.",
        reference_wav=ref,
    )
    assert len(out) > 1000
