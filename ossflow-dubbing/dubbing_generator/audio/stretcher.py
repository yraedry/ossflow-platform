"""High-quality time-stretching preserving pitch."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def trim_silence(
    audio: AudioSegment,
    threshold_db: float = -40.0,
    chunk_ms: int = 10,
) -> AudioSegment:
    """Remove leading and trailing silence from *audio*."""

    def _leading_silence(sound: AudioSegment) -> int:
        ms = 0
        while ms < len(sound) and sound[ms : ms + chunk_ms].dBFS < threshold_db:
            ms += chunk_ms
        return ms

    start = _leading_silence(audio)
    end = _leading_silence(audio.reverse())
    duration = len(audio)

    if start + end >= duration:
        return audio

    return audio[start : duration - end]


def _atempo_chain(ratio: float) -> str:
    """Build ffmpeg atempo filter chain.

    atempo accepts [0.5, 2.0]. For ratios outside this range, chain multiple
    atempo filters (e.g. 0.4x = atempo=0.5,atempo=0.8).
    """
    filters: list[str] = []
    r = ratio

    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r /= 0.5

    # Final step (if needed and not already exact)
    if abs(r - 1.0) > 0.001:
        filters.append(f"atempo={r:.4f}")

    return ",".join(filters) if filters else "atempo=1.0"


def _rubberband_stretch(
    input_wav: str, output_wav: str, ratio: float,
) -> bool:
    """Try rubberband (higher quality). Returns True if successful."""
    try:
        # rubberband: --tempo NEW/OLD. Our ratio = current/target, tempo = 1/ratio... no.
        # Actually: if we want output shorter (speed up), we need tempo > 1.
        # Our ratio = current_ms / target_ms. If ratio > 1 we need to speed up → tempo = ratio.
        tempo = ratio
        subprocess.run(
            [
                "rubberband", "--tempo", f"{tempo:.4f}",
                "--crisp", "6",  # transient handling
                input_wav, output_wav,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return os.path.exists(output_wav) and os.path.getsize(output_wav) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _atempo_ffmpeg(input_wav: str, output_wav: str, ratio: float) -> None:
    """Apply atempo chain via ffmpeg (pitch-preserving)."""
    filter_chain = _atempo_chain(ratio)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", input_wav,
            "-filter:a", filter_chain,
            "-vn", output_wav,
        ],
        check=True,
        timeout=120,
    )


def stretch_audio(
    audio: AudioSegment,
    target_duration_ms: int,
    max_ratio: float = 1.25,
    min_ratio: float = 0.5,
) -> AudioSegment:
    """Adjust *audio* to fit *target_duration_ms* preserving pitch.

    Quality strategy:
      1. Trim leading/trailing silence (free time).
      2. Clamp ratio to [min_ratio, max_ratio]. For firm quality, max 1.25x.
      3. Try rubberband first (high quality), fallback to ffmpeg atempo.
      4. If already close to target (±5%), skip processing (avoid artifacts).
    """
    trimmed = trim_silence(audio)
    current_ms = len(trimmed)

    if current_ms == 0 or target_duration_ms <= 0:
        return trimmed

    ratio = current_ms / target_duration_ms
    effective_ratio = max(min_ratio, min(ratio, max_ratio))

    # Skip tiny adjustments (avoid artifacts for imperceptible changes)
    if 0.96 <= effective_ratio <= 1.04:
        return trimmed

    tmp_dir = tempfile.mkdtemp(prefix="dubstretch_")
    tmp_in = os.path.join(tmp_dir, "in.wav")
    tmp_out = os.path.join(tmp_dir, "out.wav")

    try:
        trimmed.export(tmp_in, format="wav")

        # Prefer rubberband for quality; fallback to atempo
        ok = _rubberband_stretch(tmp_in, tmp_out, effective_ratio)
        if not ok:
            _atempo_ffmpeg(tmp_in, tmp_out, effective_ratio)

        if os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0:
            return AudioSegment.from_wav(tmp_out)
        return trimmed
    except Exception as exc:
        logger.warning("Stretch failed (%s); returning trimmed audio", exc)
        return trimmed
    finally:
        # shutil.rmtree en vez de rmdir: si ffmpeg/rubberband dejaron
        # archivos adicionales (logs, artefactos), rmdir fallaba en
        # silencio y el tempdir quedaba huérfano. Acumulado en jobs
        # largos (100+ frases) llenaba /tmp.
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning(
                "Failed to remove stretcher tempdir %s: %s", tmp_dir, exc,
            )
