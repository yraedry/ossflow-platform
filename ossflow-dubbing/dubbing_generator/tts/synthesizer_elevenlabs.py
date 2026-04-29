"""ElevenLabs cloud TTS wrapper.

Alternative to ``SynthesizerXTTSv2`` that offloads synthesis to ElevenLabs.
The voice must be pre-registered in the ElevenLabs dashboard (PVC or IVC) —
``reference_wav`` is accepted for API parity with the XTTS backend but
ignored here (the cloned speaker lives on the provider).

Chosen over XTTS when local quality plateaus: XTTS-v2 hits ~MOS 1.5 on our
BJJ content (flat prosody, robotic cold-starts, no real speed control).
ElevenLabs multilingual_v2 respects SRT slots far better and needs no
demucs/stretch-nudge rescue passes — though it costs money and needs
network. Swap via ``cfg.tts_engine = "elevenlabs"``.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

from pydub import AudioSegment

from ..config import DubbingConfig

logger = logging.getLogger(__name__)

_PCM_SAMPLE_RATES = {
    "pcm_16000": 16000,
    "pcm_22050": 22050,
    "pcm_24000": 24000,
    "pcm_44100": 44100,
    "pcm_48000": 48000,
}


class SynthesizerElevenLabs:
    """Generate speech from text using ElevenLabs.

    The client is created lazily so importing the module does not require
    the API key to be set (tests can construct the object with a mock).
    """

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config
        self._client = None
        self._voice_settings = None
        self._sample_rate = _PCM_SAMPLE_RATES.get(
            config.elevenlabs_output_format, 24000
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get(self.cfg.elevenlabs_api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{self.cfg.elevenlabs_api_key_env} is empty. "
                f"Set it in the dubbing-generator container environment."
            )
        from elevenlabs.client import ElevenLabs
        self._client = ElevenLabs(
            api_key=api_key,
            timeout=self.cfg.elevenlabs_request_timeout,
        )
        logger.info(
            "ElevenLabs client ready (voice_id=%s, model=%s, format=%s)",
            self.cfg.elevenlabs_voice_id,
            self.cfg.elevenlabs_model_id,
            self.cfg.elevenlabs_output_format,
        )
        return self._client

    def _get_voice_settings(self):
        if self._voice_settings is not None:
            return self._voice_settings
        from elevenlabs import VoiceSettings
        self._voice_settings = VoiceSettings(
            stability=self.cfg.elevenlabs_stability,
            similarity_boost=self.cfg.elevenlabs_similarity_boost,
            style=self.cfg.elevenlabs_style,
            use_speaker_boost=self.cfg.elevenlabs_use_speaker_boost,
        )
        return self._voice_settings

    def generate(
        self,
        text: str,
        reference_wav: Path,
        speed: float | None = None,
    ) -> AudioSegment:
        """Synthesize *text*.

        ``reference_wav`` is accepted for interface parity with the XTTS
        backend but unused — the voice clone lives on ElevenLabs servers
        keyed by ``cfg.elevenlabs_voice_id``. ``speed`` is also accepted
        for parity; ElevenLabs v2 does not expose per-request speed, so
        the pipeline's time-stretcher handles slot fitting as usual.
        """
        _ = reference_wav
        _ = speed

        if not text.strip():
            return AudioSegment.silent(duration=100)

        client = self._get_client()
        voice_settings = self._get_voice_settings()

        try:
            audio_iter = client.text_to_speech.convert(
                voice_id=self.cfg.elevenlabs_voice_id,
                text=text,
                model_id=self.cfg.elevenlabs_model_id,
                output_format=self.cfg.elevenlabs_output_format,
                voice_settings=voice_settings,
            )
        except Exception as exc:
            logger.warning(
                "ElevenLabs convert failed (text=%r): %s",
                text[:80], exc,
            )
            return AudioSegment.silent(duration=200)

        raw = _collect_audio_bytes(audio_iter)
        if not raw:
            logger.warning("ElevenLabs returned empty audio for text=%r", text[:80])
            return AudioSegment.silent(duration=200)

        return _decode_audio(raw, self.cfg.elevenlabs_output_format, self._sample_rate)


def _collect_audio_bytes(audio_iter) -> bytes:
    """Handle both iterator-of-bytes and raw-bytes SDK returns."""
    if isinstance(audio_iter, (bytes, bytearray)):
        return bytes(audio_iter)
    buf = bytearray()
    for chunk in audio_iter:
        if chunk:
            buf.extend(chunk)
    return bytes(buf)


def _decode_audio(raw: bytes, output_format: str, sample_rate: int) -> AudioSegment:
    """Convert raw PCM/MP3 bytes into an AudioSegment."""
    if output_format.startswith("pcm_"):
        return AudioSegment(
            data=raw,
            sample_width=2,      # pcm_s16le
            frame_rate=sample_rate,
            channels=1,
        )
    return AudioSegment.from_file(io.BytesIO(raw))
