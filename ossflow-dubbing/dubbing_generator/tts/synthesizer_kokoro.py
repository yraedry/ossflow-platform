"""Kokoro-82M local TTS backend.

StyleTTS 2-based open source TTS (Apache 2.0). Spanish male voice
``em_alex`` (or ``em_santa``) — preset, no voice cloning. Output is 24 kHz
mono native, matches the pipeline sample rate without resampling.

Trade-off vs ElevenLabs: not the same speaker as the instructor, but
naturally-sounding masculine ES voice, free, GPU-accelerated.
Trade-off vs Piper: better prosody (StyleTTS 2 > VITS), slightly
slower inference, requires GPU for reasonable speed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from ..config import DubbingConfig

logger = logging.getLogger(__name__)

_TARGET_SAMPLE_RATE = 24000


class SynthesizerKokoro:
    """Generate speech from text using Kokoro-82M (local)."""

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config
        self._pipeline = None

    @property
    def sample_rate(self) -> int:
        return _TARGET_SAMPLE_RATE

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code=self.cfg.kokoro_lang_code)
        logger.info(
            "Kokoro pipeline ready (lang=%s, voice=%s, speed=%.2f)",
            self.cfg.kokoro_lang_code,
            self.cfg.kokoro_voice,
            self.cfg.kokoro_speed,
        )
        return self._pipeline

    def generate(
        self,
        text: str,
        reference_wav: Path,
        speed: float | None = None,
    ) -> AudioSegment:
        """Synthesize *text*.

        ``reference_wav`` is accepted for interface parity but unused —
        Kokoro does not do voice cloning. ``speed`` overrides
        ``cfg.kokoro_speed`` if provided.
        """
        _ = reference_wav

        if not text.strip():
            return AudioSegment.silent(duration=100)

        pipeline = self._get_pipeline()
        effective_speed = speed if (speed and speed > 0) else self.cfg.kokoro_speed

        try:
            chunks = []
            for _gs, _ps, audio in pipeline(
                text,
                voice=self.cfg.kokoro_voice,
                speed=effective_speed,
            ):
                chunks.append(audio)
        except Exception as exc:
            logger.warning(
                "Kokoro synthesize failed (text=%r): %s",
                text[:80], exc,
            )
            return AudioSegment.silent(duration=200)

        if not chunks:
            return AudioSegment.silent(duration=200)

        full = np.concatenate(chunks).astype(np.float32)
        # Kokoro returns float32 in [-1, 1]; convert to int16 PCM.
        pcm16 = np.clip(full * 32767, -32768, 32767).astype(np.int16)

        segment = AudioSegment(
            data=pcm16.tobytes(),
            sample_width=2,
            frame_rate=_TARGET_SAMPLE_RATE,
            channels=1,
        )
        return segment
