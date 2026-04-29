"""Voice detection using Demucs vocal separation with hysteresis."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from demucs.apply import apply_model
from demucs.pretrained import get_model

from ..config import Config

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Voice detection using Demucs. Extracts audio via ffmpeg, not moviepy.

    Uses hysteresis thresholds to avoid rapid toggling between voice/silence:
    - voice_enter_threshold (0.30): RMS must exceed this to declare voice present
    - voice_exit_threshold  (0.20): RMS must drop below this to declare silence
    - Intermediate zone (0.20-0.30) retains the previous state (UNCERTAIN)

    Analysis uses fine-grained windows (100ms) with moving-average smoothing
    (500ms) for stable detection.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.separator = get_model("htdemucs")
            self.separator.to(self.device)
            logger.info("Demucs loaded on %s", self.device)
        except Exception as e:
            logger.error("Failed to load Demucs: %s", e)
            raise

        # Fine-grained voice map: key = window index, value = smoothed RMS
        self._voice_map: dict[int, float] = {}
        # Hysteresis state map: True = voice, False = silence, None = unknown
        self._voice_state: dict[int, Optional[bool]] = {}
        self._sample_rate: int = 44100
        self._window_samples: int = 0
        self._smoothing_windows: int = 0

        # Transition/sting statistics -- filled by precompute.
        self._rms_mean: float = 0.0
        self._rms_std: float = 0.0
        # Sorted list of timestamps (seconds) where an RMS "sting" peak occurred.
        self._transition_peaks: list[float] = []

    def _extract_chunk_audio(self, video_path: str, start: float,
                             duration: float) -> Optional[Path]:
        """Extract an audio chunk to a temp WAV file using ffmpeg."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", video_path,
            "-t", f"{duration:.3f}",
            "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le",
            str(tmp_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return tmp_path
        except subprocess.CalledProcessError as e:
            logger.debug("ffmpeg audio extraction failed: %s",
                         e.stderr[:200] if e.stderr else "")
            tmp_path.unlink(missing_ok=True)
            return None

    def _process_chunk(self, wav_path: Path) -> np.ndarray:
        """Run Demucs on a WAV file, return per-window vocal RMS array.

        Windows are voice_window_ms (default 100ms) with moving-average
        smoothing over voice_smoothing_ms (default 500ms).
        """
        wav, sr = torchaudio.load(str(wav_path))
        if sr != self._sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self._sample_rate)
            sr = self._sample_rate

        wav = wav.to(self.device)
        # Normalise
        ref = wav.mean(0)
        wav = wav - ref.mean()
        std = ref.std() + 1e-8
        wav = wav / std
        wav = wav.unsqueeze(0)

        with torch.no_grad():
            sources = apply_model(self.separator, wav, shifts=0, split=True,
                                  overlap=0.25, progress=False)
        vocals = sources[0, -1]  # vocals track
        vocals_np = vocals.cpu().numpy().flatten()

        # Compute per-window RMS (window = voice_window_ms)
        window_samples = self._window_samples
        n_windows = max(1, len(vocals_np) // window_samples)
        rms_values = np.zeros(n_windows)
        for i in range(n_windows):
            chunk = vocals_np[i * window_samples:(i + 1) * window_samples]
            rms_values[i] = float(np.sqrt(np.mean(chunk ** 2)))

        # Apply moving-average smoothing
        smoothing_w = self._smoothing_windows
        if smoothing_w > 1 and len(rms_values) > 1:
            kernel = np.ones(smoothing_w) / smoothing_w
            # Pad to avoid edge shrinkage
            padded = np.pad(rms_values, (smoothing_w // 2, smoothing_w // 2),
                            mode='edge')
            rms_values = np.convolve(padded, kernel, mode='valid')
            # Ensure length matches
            rms_values = rms_values[:n_windows]

        return rms_values

    def _apply_hysteresis(self, start_window: int, rms_array: np.ndarray) -> None:
        """Apply hysteresis thresholds to RMS array and update state map."""
        enter_thresh = self.cfg.voice_enter_threshold
        exit_thresh = self.cfg.voice_exit_threshold

        for i, rms in enumerate(rms_array):
            idx = start_window + i
            self._voice_map[idx] = float(rms)

            if rms >= enter_thresh:
                self._voice_state[idx] = True  # voice
            elif rms <= exit_thresh:
                self._voice_state[idx] = False  # silence
            else:
                # Hysteresis zone: carry forward previous state
                prev_state = self._voice_state.get(idx - 1)
                self._voice_state[idx] = prev_state  # None if no previous

    def precompute(self, video_path: str, duration: float) -> None:
        """Build the full voice map for the video in 10-second chunks."""
        self._voice_map.clear()
        self._voice_state.clear()

        # Compute window parameters
        sr = self._sample_rate
        self._window_samples = max(1, int(sr * self.cfg.voice_window_ms / 1000))
        self._smoothing_windows = max(
            1, self.cfg.voice_smoothing_ms // self.cfg.voice_window_ms
        )

        window_duration = self.cfg.voice_window_ms / 1000.0  # seconds per window

        chunk_dur = self.cfg.audio_chunk_seconds
        t = 0.0
        total_chunks = int(duration / chunk_dur) + 1
        logger.info("Pre-computing voice map (%d chunks of %.0fs, "
                     "window=%dms, smoothing=%dms)...",
                     total_chunks, chunk_dur,
                     self.cfg.voice_window_ms, self.cfg.voice_smoothing_ms)

        chunk_idx = 0
        while t < duration:
            chunk_idx += 1
            actual_dur = min(chunk_dur, duration - t)
            if actual_dur < 0.5:
                break

            # Calculate the starting window index for this chunk
            start_window = int(t / window_duration)

            wav_path = self._extract_chunk_audio(video_path, t, actual_dur)
            if wav_path is None:
                # Mark these windows as unknown (NOT voiced -- don't assume)
                n_windows = max(1, int(actual_dur / window_duration))
                for w in range(n_windows):
                    idx = start_window + w
                    self._voice_map[idx] = -1.0  # sentinel for unknown
                    self._voice_state[idx] = None
                t += chunk_dur
                continue

            try:
                rms_arr = self._process_chunk(wav_path)
                self._apply_hysteresis(start_window, rms_arr)
            except Exception as e:
                logger.debug("Demucs chunk failed at %.1f: %s", t, e)
                # Mark as unknown, not voiced
                n_windows = max(1, int(actual_dur / window_duration))
                for w in range(n_windows):
                    idx = start_window + w
                    self._voice_map[idx] = -1.0
                    self._voice_state[idx] = None
            finally:
                wav_path.unlink(missing_ok=True)

            if chunk_idx % 20 == 0:
                pct = min(100, int(t / duration * 100))
                logger.info("  Voice map: %d%% (%d/%d chunks)", pct,
                            chunk_idx, total_chunks)
            t += chunk_dur

        logger.info("Voice map complete: %d windows mapped.", len(self._voice_map))

        # Compute global RMS statistics and transition peaks (musical stings).
        self._compute_transition_stats()

    def _compute_transition_stats(self) -> None:
        """Derive mean/std of RMS and locate short high-RMS peaks (stings).

        A "sting" is a window index where RMS exceeds mean + 2 sigma AND the
        surrounding context (+-3s) is relatively quiet. We only keep the
        timestamps (seconds) for downstream corroboration.
        """
        self._transition_peaks = []
        if not self._voice_map:
            self._rms_mean = 0.0
            self._rms_std = 0.0
            return

        valid = np.array(
            [v for v in self._voice_map.values() if v >= 0],
            dtype=np.float32,
        )
        if valid.size == 0:
            self._rms_mean = 0.0
            self._rms_std = 0.0
            return

        self._rms_mean = float(valid.mean())
        self._rms_std = float(valid.std())
        peak_thresh = self._rms_mean + 2.0 * self._rms_std

        window_duration = self.cfg.voice_window_ms / 1000.0
        for idx, rms in self._voice_map.items():
            if rms < 0:
                continue
            if rms >= peak_thresh:
                self._transition_peaks.append(idx * window_duration)

        self._transition_peaks.sort()
        logger.info(
            "Transition stats: mean=%.3f std=%.3f peaks=%d",
            self._rms_mean, self._rms_std, len(self._transition_peaks),
        )

    def is_transition_window(self, t: float, window_s: float = 2.0) -> bool:
        """True if an RMS "sting" peak lies within [t-window_s, t+window_s]."""
        if not self._transition_peaks:
            return False
        lo = t - window_s
        hi = t + window_s
        # Linear scan is fine -- O(N) peaks is small (hundreds at most).
        for pk in self._transition_peaks:
            if pk < lo:
                continue
            if pk > hi:
                break
            return True
        return False

    def _second_to_window(self, second: int) -> int:
        """Convert a second to the corresponding window index."""
        window_duration = self.cfg.voice_window_ms / 1000.0
        return int(second / window_duration)

    def voice_level(self, second: int) -> float:
        """Return the vocal RMS for a given second (0-based).

        Returns the average RMS across all windows within that second.
        Returns -1.0 if the second was not analyzed (unknown).
        """
        window_duration = self.cfg.voice_window_ms / 1000.0
        windows_per_sec = max(1, int(1.0 / window_duration))
        start_window = self._second_to_window(second)

        values = []
        for w in range(windows_per_sec):
            idx = start_window + w
            rms = self._voice_map.get(idx)
            if rms is not None and rms >= 0:
                values.append(rms)

        if not values:
            return -1.0  # unknown
        return float(np.mean(values))

    def get_voice_confidence(self, second: int) -> float:
        """Return a confidence value 0.0-1.0 for voice presence at this second.

        0.0 = definitely silence, 1.0 = definitely voice.
        Returns 0.5 (uncertain) if the second could not be analyzed.
        """
        level = self.voice_level(second)
        if level < 0:
            return 0.5  # unknown -> uncertain

        # Normalize RMS to 0-1 range using thresholds
        exit_t = self.cfg.voice_exit_threshold
        enter_t = self.cfg.voice_enter_threshold

        if level <= exit_t:
            # Scale 0..exit_t to 0..0.3
            return min(0.3, level / max(exit_t, 1e-8) * 0.3)
        elif level >= enter_t:
            # Scale enter_t..1.0 to 0.7..1.0
            above = level - enter_t
            scale = min(1.0, above / max(1.0 - enter_t, 1e-8))
            return 0.7 + scale * 0.3
        else:
            # Hysteresis zone: 0.3..0.7
            ratio = (level - exit_t) / max(enter_t - exit_t, 1e-8)
            return 0.3 + ratio * 0.4

    def is_silent(self, second: int) -> bool:
        """Return True if the second is classified as silence via hysteresis."""
        window_duration = self.cfg.voice_window_ms / 1000.0
        windows_per_sec = max(1, int(1.0 / window_duration))
        start_window = self._second_to_window(second)

        voice_count = 0
        silence_count = 0
        unknown_count = 0

        for w in range(windows_per_sec):
            idx = start_window + w
            state = self._voice_state.get(idx)
            if state is True:
                voice_count += 1
            elif state is False:
                silence_count += 1
            else:
                unknown_count += 1

        total = voice_count + silence_count + unknown_count
        if total == 0:
            return False  # no data -> conservative: not silent

        # Majority vote across windows in this second
        # If mostly silent, return True
        return silence_count > voice_count

    def is_uncertain(self, second: int) -> bool:
        """Return True if the second is in the hysteresis zone (uncertain)."""
        window_duration = self.cfg.voice_window_ms / 1000.0
        windows_per_sec = max(1, int(1.0 / window_duration))
        start_window = self._second_to_window(second)

        unknown_count = 0
        total = 0

        for w in range(windows_per_sec):
            idx = start_window + w
            state = self._voice_state.get(idx)
            total += 1
            if state is None:
                unknown_count += 1

        if total == 0:
            return True
        # Uncertain if majority of windows are in hysteresis/unknown zone
        return unknown_count > total // 2
