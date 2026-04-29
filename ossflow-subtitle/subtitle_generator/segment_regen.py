"""Regenerate a single subtitle segment by re-transcribing its audio window."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .config import EXTENSIONES, TranscriptionConfig, DEFAULT_HOTWORDS

log = logging.getLogger("subtitler")


def find_sibling_video(srt_path: Path) -> Optional[Path]:
    """Locate a video file sharing the SRT's basename in the same folder."""
    srt_path = Path(srt_path)
    stem = srt_path.stem
    parent = srt_path.parent
    for ext in EXTENSIONES:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def extract_audio_window(
    video_path: Path,
    start: float,
    end: float,
    context_seconds: float = 1.0,
) -> Path:
    """Extract [start-ctx, end+ctx] of *video_path* to a temp 16kHz mono WAV.

    Returns the temp WAV path — caller is responsible for deletion.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg no está disponible en el PATH")

    win_start = max(0.0, start - context_seconds)
    duration = max(0.1, (end - start) + 2 * context_seconds)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out = Path(tmp.name)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{win_start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-ac", "1", "-ar", "16000",
        "-vn", str(out),
    ]
    # Timeout defensivo: ffmpeg sin timeout puede colgar el worker
    # FastAPI indefinidamente si el vídeo tiene codec exótico o el
    # filesystem está lento (NAS, I/O stall). 60 s es muy holgado
    # para extraer una ventana de <30 s — si excede, algo va mal.
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60.0,
        )
    except subprocess.TimeoutExpired:
        out.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg timeout (>60s) extracting window from {video_path.name}",
        )
    if proc.returncode != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg falló: {proc.stderr.strip()}")
    return out


class SegmentRegenerator:
    """Re-transcribe a single time window using an already-loaded WhisperX model."""

    def __init__(self, t_config: TranscriptionConfig) -> None:
        self.t_config = t_config
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import whisperx
        log.info("SegmentRegenerator loading WhisperX %s on %s",
                 self.t_config.model_name, self.t_config.device)
        self._model = whisperx.load_model(
            self.t_config.model_name,
            self.t_config.device,
            compute_type=self.t_config.compute_type,
            asr_options={
                "initial_prompt": self.t_config.initial_prompt,
                "hotwords": self.t_config.hotwords if self.t_config.hotwords is not None else DEFAULT_HOTWORDS,
                "beam_size": self.t_config.beam_size,
                "condition_on_previous_text": False,
            },
            vad_options={
                "vad_onset": self.t_config.vad_onset,
                "vad_offset": self.t_config.vad_offset,
            },
        )

    def regenerate(
        self,
        video_path: Path,
        start: float,
        end: float,
        context_seconds: float = 1.0,
    ) -> dict:
        """Return {text, start, end} for the re-transcribed window."""
        import whisperx
        self.load()

        wav_path = extract_audio_window(video_path, start, end, context_seconds)
        try:
            audio = whisperx.load_audio(str(wav_path))
            result = self._model.transcribe(
                audio,
                batch_size=self.t_config.batch_size,
                language=self.t_config.language,
            )
        finally:
            wav_path.unlink(missing_ok=True)

        segs = result.get("segments", []) or []
        if not segs:
            return {"text": "", "start": start, "end": end}

        text_parts = [(s.get("text") or "").strip() for s in segs]
        text = " ".join(p for p in text_parts if p).strip()

        # Offsets in the extracted window are relative; shift back to absolute.
        win_start_abs = max(0.0, start - context_seconds)
        abs_start = win_start_abs + float(segs[0].get("start", 0.0) or 0.0)
        abs_end = win_start_abs + float(segs[-1].get("end", 0.0) or 0.0)

        if abs_end <= abs_start:
            abs_start, abs_end = start, end

        return {"text": text, "start": abs_start, "end": abs_end}
