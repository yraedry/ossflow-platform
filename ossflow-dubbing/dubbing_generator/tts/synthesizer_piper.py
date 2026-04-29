"""Piper TTS local backend.

Local Spanish voice synthesis via Piper (CPU, ONNX). No voice cloning —
the voice is chosen from the Piper voice catalog (``es_ES-sharvard-medium``
by default). Free, fast, and deterministic; trade-off vs ElevenLabs is
flat prosody and English-word mispronunciation. Mitigated by
``bjj_casting.castellanize`` which runs in the pipeline before reaching
this synthesizer.

Output is upsampled from Piper's native rate (22050 Hz for medium voices)
to 24000 Hz so the rest of the audio stack (mixer, stretcher, demucs)
keeps the same sample rate as ElevenLabs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydub import AudioSegment

from ..config import DubbingConfig

logger = logging.getLogger(__name__)

_TARGET_SAMPLE_RATE = 24000


class SynthesizerPiper:
    """Generate speech from text using Piper (local ONNX runtime)."""

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config
        self._voice = None

    @property
    def sample_rate(self) -> int:
        return _TARGET_SAMPLE_RATE

    def _get_voice(self):
        if self._voice is not None:
            return self._voice
        from piper import PiperVoice

        model_path = Path(self.cfg.piper_model_path)
        if not model_path.exists():
            raise RuntimeError(
                f"Piper model not found at {model_path}. "
                f"Build the dubbing-generator image with the model baked in, "
                f"or mount it at this path."
            )
        self._voice = PiperVoice.load(str(model_path))
        logger.info(
            "Piper voice ready (model=%s, length_scale=%.2f, noise_scale=%.2f, noise_w=%.2f)",
            model_path.name,
            self.cfg.piper_length_scale,
            self.cfg.piper_noise_scale,
            self.cfg.piper_noise_w,
        )
        return self._voice

    def _build_syn_config(self, speed: float | None):
        from piper.config import SynthesisConfig

        length_scale = self.cfg.piper_length_scale
        if speed and speed > 0:
            length_scale = self.cfg.piper_length_scale / speed
        return SynthesisConfig(
            length_scale=length_scale,
            noise_scale=self.cfg.piper_noise_scale,
            noise_w_scale=self.cfg.piper_noise_w,
        )

    def generate(
        self,
        text: str,
        reference_wav: Path,
        speed: float | None = None,
    ) -> AudioSegment:
        """Synthesize *text*.

        ``reference_wav`` is accepted for interface parity but unused —
        Piper does not do voice cloning. ``speed`` is accepted for parity
        and converted to ``length_scale`` (inverse: speed>1 → shorter
        utterance → length_scale<1).
        """
        _ = reference_wav

        if not text.strip():
            return AudioSegment.silent(duration=100)

        voice = self._get_voice()
        syn_config = self._build_syn_config(speed)

        try:
            chunks = list(voice.synthesize(text, syn_config=syn_config))
        except Exception as exc:
            logger.warning(
                "Piper synthesize failed (text=%r): %s",
                text[:80], exc,
            )
            return AudioSegment.silent(duration=200)

        if not chunks:
            return AudioSegment.silent(duration=200)

        first = chunks[0]
        raw = b"".join(c.audio_int16_bytes for c in chunks)

        segment = AudioSegment(
            data=raw,
            sample_width=first.sample_width,
            frame_rate=first.sample_rate,
            channels=first.sample_channels,
        )
        if segment.frame_rate != _TARGET_SAMPLE_RATE:
            segment = segment.set_frame_rate(_TARGET_SAMPLE_RATE)
        if segment.channels != 1:
            segment = segment.set_channels(1)
        if segment.sample_width != 2:
            segment = segment.set_sample_width(2)
        return segment
