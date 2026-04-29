"""Voice reference extraction with VAD-based segment selection."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..config import DubbingConfig

logger = logging.getLogger(__name__)


class VoiceCloner:
    """Extract the best voice reference sample from a video.

    Uses Silero VAD to find the segment with the most speech,
    then extracts ``voice_sample_duration`` seconds (default 30 s)
    at 24 kHz mono for optimal XTTS v2 cloning.
    """

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_reference(
        self,
        video_path: Path,
        override_path: Optional[Path] = None,
    ) -> Path:
        """Return the path to a voice reference WAV.

        Priority:
          1. *override_path* if given (e.g. Demucs vocals stem, clean speech).
          2. Model voice if ``use_model_voice`` is True.
          3. Extracted from *video_path* using VAD to pick best segment.
        """
        # Model voice takes priority: if the operator configured a native ES
        # voice WAV, use it. Cloning the instructor's EN voice yields an ES
        # output with strong English accent and prosody; a native ES ref fixes
        # both, at the cost of losing the instructor's original timbre.
        if self.cfg.use_model_voice and self.cfg.model_voice_path:
            model_path = Path(self.cfg.model_voice_path)
            if model_path.exists():
                logger.info("Using model voice (native ES): %s", model_path)
                return model_path
            logger.warning(
                "use_model_voice set but model_voice_path missing: %s",
                self.cfg.model_voice_path,
            )

        if override_path and override_path.exists():
            # If override is the vocals stem, pick best VAD window and resample
            # to 24kHz mono for XTTS (same treatment as raw video extraction).
            logger.info("Using override voice profile: %s", override_path)
            refined = override_path.with_name(f"{video_path.stem}_ref.wav")
            if refined.exists():
                return refined
            try:
                start, duration = self._find_best_speech_segment(override_path)
                self._extract_audio(override_path, start, duration, refined)
                return refined
            except Exception as exc:
                logger.warning("Refining vocals stem failed (%s); using raw stem", exc)
                return override_path

        # Extract from video
        output_wav = video_path.with_name(f"{video_path.stem}_ref.wav")
        if output_wav.exists():
            logger.info("Voice reference already exists: %s", output_wav)
            return output_wav

        logger.info("Extracting voice reference from %s ...", video_path.name)
        start, duration = self._find_best_speech_segment(video_path)
        self._extract_audio(video_path, start, duration, output_wav)
        return output_wav

    # ------------------------------------------------------------------
    # VAD-based segment selection
    # ------------------------------------------------------------------

    def _find_best_speech_segment(
        self, video_path: Path,
    ) -> tuple[float, float]:
        """Use Silero VAD to find the segment with most speech.

        Returns ``(start_seconds, duration_seconds)``.
        Falls back to t=60 s if VAD fails.
        """
        duration = self.cfg.voice_sample_duration

        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            (get_speech_timestamps, _, read_audio, *_) = utils

            # Extract first 5 minutes of audio for VAD analysis
            analysis_duration = 300.0
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", prefix="vad_", delete=False,
            )
            tmp.close()
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(video_path),
                        "-t", str(analysis_duration),
                        "-vn", "-acodec", "pcm_s16le",
                        "-ar", "16000", "-ac", "1",
                        tmp.name,
                    ],
                    check=True,
                )
                wav = read_audio(tmp.name, sampling_rate=16000)
            finally:
                try:
                    Path(tmp.name).unlink(missing_ok=True)
                except OSError:
                    pass

            timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

            if not timestamps:
                logger.warning("VAD found no speech; falling back to t=60 s")
                return (60.0, duration)

            # Sliding window: find the window of *duration* seconds with
            # the most speech frames
            window_samples = int(duration * 16000)
            total_samples = len(wav)
            best_start = 0
            best_speech = 0.0

            step = int(5 * 16000)  # 5-second steps
            for win_start in range(0, total_samples - window_samples, step):
                win_end = win_start + window_samples
                speech_in_window = sum(
                    min(ts["end"], win_end) - max(ts["start"], win_start)
                    for ts in timestamps
                    if ts["end"] > win_start and ts["start"] < win_end
                )
                if speech_in_window > best_speech:
                    best_speech = speech_in_window
                    best_start = win_start

            start_sec = best_start / 16000.0
            logger.info(
                "Best speech segment: %.1f s (%.1f s of speech in window)",
                start_sec,
                best_speech / 16000.0,
            )
            return (start_sec, duration)

        except Exception as exc:
            logger.warning("VAD failed (%s); falling back to t=60 s", exc)
            return (60.0, duration)

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------

    def _extract_audio(
        self,
        video_path: Path,
        start: float,
        duration: float,
        output: Path,
    ) -> None:
        """Extract audio at 24 kHz mono with voice-cleanup filters.

        Filters applied:
          - highpass=80: remove low-frequency rumble
          - lowpass=8000: remove very high frequencies (focus on voice band)
          - loudnorm: normalize perceived loudness (EBU R128)
        """
        af_chain = "highpass=f=80,lowpass=f=8000,loudnorm=I=-18:TP=-2:LRA=7"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-vn",
            "-af", af_chain,
            "-acodec", "pcm_s16le",
            "-ar", "24000", "-ac", "1",
            str(output),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            logger.warning("Extraction at t=%.1f failed; trying t=0", start)
            cmd[4] = "0"
            subprocess.run(cmd, check=True)

        logger.info("Voice reference saved: %s", output)
