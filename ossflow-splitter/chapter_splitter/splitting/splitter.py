"""Video splitting with ffmpeg and NVENC encoding."""

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from ..config import Config
from ..models import Chapter

logger = logging.getLogger(__name__)


class VideoSplitter:
    """Precise ffmpeg splitting with two-pass seeking and NVENC encoding."""

    def __init__(self, config: Config) -> None:
        self.cfg = config

    def split(self, video_path: str, chapters: list[Chapter],
              output_dir: Path, show_name: str, season: int,
              dry_run: bool = False) -> None:
        """Split video into chapter files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating %d chapter files...", len(chapters))

        for idx, ch in enumerate(chapters, start=1):
            title = ch.title[:100] if len(ch.title) > 100 else ch.title
            if len(title) < 3:
                title = f"Technique {idx}"

            filename = f"{show_name} - S{season:02d}E{idx:02d} - {title}{self.cfg.output_format}"
            filename = re.sub(r'[\\/*?:"<>|]', "", filename)
            out_path = output_dir / filename

            if out_path.exists():
                logger.info("  Skip (exists): %s", filename)
                continue

            if dry_run:
                end_str = f"{ch.end:.1f}" if ch.end is not None else "EOF"
                logger.info("  [DRY RUN] Would create: %s  (%.1f -> %s)",
                            filename, ch.start, end_str)
                continue

            cmd = self._build_cmd(video_path, str(out_path), ch.start, ch.end)
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info("  OK: %s", filename)
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode(errors="replace")[:300] if e.stderr else ""
                logger.error("  ffmpeg failed: %s -- %s", filename, stderr)

    def _build_cmd(self, input_path: str, output_path: str,
                   start: float, end: Optional[float]) -> list[str]:
        """Build ffmpeg command with two-pass seeking."""
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

        # Two-pass seek: fast seek to 30s before, then precise offset
        fast_seek = max(0.0, start - 30.0)
        precise_offset = start - fast_seek

        cmd.extend(["-ss", f"{fast_seek:.3f}"])
        cmd.extend(["-i", input_path])
        cmd.extend(["-ss", f"{precise_offset:.3f}"])

        if end is not None:
            duration = end - start
            cmd.extend(["-t", f"{duration:.3f}"])

        cmd.extend([
            "-c:v", self.cfg.encoder,
            "-preset", self.cfg.preset,
            "-cq", str(self.cfg.cq),
            "-c:a", self.cfg.audio_codec,
            output_path,
        ])
        return cmd
