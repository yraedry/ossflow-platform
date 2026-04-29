"""Oracle-driven splitter: cuts mp4s using BJJFanatics-scraped timestamps.

Unlike :class:`VideoSplitter` (signal-based), this module trusts the oracle
absolutely and uses ``ffmpeg -c copy`` for keyframe-aligned fast splits.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..oracle.models import OracleResult, OracleVolume
from ..utils import sanitize_filename

logger = logging.getLogger(__name__)

ProgressCb = Callable[[float, str], None]

DURATION_TOLERANCE_S = 5.0
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov")
MAX_VOLUME_NUMBER = 50

# Regex extracting trailing number from a stem, optionally separated by space.
_STEM_NUMBER_RE = re.compile(r"(\d+)\s*$")


@dataclass
class SplitReport:
    """Outcome of an :class:`OracleSplitter` run."""

    volumes_processed: int = 0
    chapters_created: int = 0
    warnings: list[str] = field(default_factory=list)
    needs_review_flags: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "volumes_processed": self.volumes_processed,
            "chapters_created": self.chapters_created,
            "warnings": list(self.warnings),
            "needs_review_flags": list(self.needs_review_flags),
        }


class OracleSplitter:
    """Cuts mp4s by oracle timestamps into ``Season NN/`` folders."""

    def __init__(
        self,
        instructional_dir: Path,
        oracle: OracleResult,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.instructional_dir = Path(instructional_dir)
        self.oracle = oracle
        self.output_dir = Path(output_dir) if output_dir else self.instructional_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def split(self, progress_cb: Optional[ProgressCb] = None) -> SplitReport:
        report = SplitReport()
        total_chapters = sum(len(v.chapters) for v in self.oracle.volumes)
        if total_chapters == 0:
            return report

        done = 0
        for volume in self.oracle.volumes:
            mp4 = self._locate_mp4_for_volume(volume.number)
            if mp4 is None:
                msg = f"No mp4 found for Volume {volume.number}, skipping"
                logger.warning(msg)
                report.warnings.append(msg)
                # Still advance progress for the chapters we are skipping so the
                # bar reaches 100%.
                done += len(volume.chapters)
                if progress_cb is not None:
                    pct = (done / total_chapters) * 100.0
                    progress_cb(pct, msg)
                continue

            actual_duration = self._probe_duration(mp4)
            if actual_duration is not None:
                diff = abs(actual_duration - volume.total_duration_s)
                if diff > DURATION_TOLERANCE_S:
                    msg = (
                        f"Volume {volume.number}: mp4 duration "
                        f"{actual_duration:.1f}s differs from oracle "
                        f"{volume.total_duration_s:.1f}s by {diff:.1f}s "
                        f"(tolerance {DURATION_TOLERANCE_S}s)"
                    )
                    logger.warning(msg)
                    report.warnings.append(msg)
                    if volume.number not in report.needs_review_flags:
                        report.needs_review_flags.append(volume.number)

            season_dir = self.output_dir / f"Season {volume.number:02d}"
            season_dir.mkdir(parents=True, exist_ok=True)

            for idx, chapter in enumerate(volume.chapters, start=1):
                title = sanitize_filename(chapter.title) or f"Chapter {idx}"
                filename = (
                    f"S{volume.number:02d}E{idx:02d} - {title}.mp4"
                )
                out_path = season_dir / filename
                ok, err = self._cut(mp4, out_path, chapter.start_s, chapter.end_s)
                done += 1
                if ok:
                    report.chapters_created += 1
                    msg = f"Created {filename}"
                else:
                    msg = f"Failed {filename}: {err}"
                    logger.error(msg)
                    report.warnings.append(msg)
                if progress_cb is not None:
                    pct = (done / total_chapters) * 100.0
                    progress_cb(pct, msg)

            report.volumes_processed += 1

        # Ensure we end at 100%.
        if progress_cb is not None and total_chapters > 0:
            progress_cb(100.0, "done")

        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _locate_mp4_for_volume(self, number: int) -> Optional[Path]:
        candidates: list[Path] = []
        for p in self.instructional_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            m = _STEM_NUMBER_RE.search(p.stem)
            if not m:
                continue
            num = int(m.group(1))
            if num > MAX_VOLUME_NUMBER:
                continue
            if num == number:
                candidates.append(p)
        if not candidates:
            return None
        # Prefer .mp4 then by name
        candidates.sort(key=lambda p: (p.suffix.lower() != ".mp4", p.name))
        return candidates[0]

    @staticmethod
    def _probe_duration(path: Path) -> Optional[float]:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return None
        try:
            res = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-print_format", "json",
                    "-show_format",
                    str(path),
                ],
                capture_output=True, text=True, timeout=15, check=False,
            )
        except Exception:
            return None
        if res.returncode != 0:
            return None
        try:
            data = json.loads(res.stdout or "{}")
            return float(data.get("format", {}).get("duration"))
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _cut(
        src: Path, dst: Path, start_s: float, end_s: float
    ) -> tuple[bool, str]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False, "ffmpeg not found in PATH"
        if dst.exists():
            return True, "exists"
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}",
            "-to", f"{end_s:.3f}",
            "-i", str(src),
            "-c", "copy",
            str(dst),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        if res.returncode != 0:
            stderr = (res.stderr or b"").decode(errors="replace")[:300]
            return False, stderr
        return True, ""
