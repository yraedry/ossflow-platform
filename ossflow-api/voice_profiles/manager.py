"""Manage voice reference samples per instructor for consistent TTS dubbing."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """A stored voice reference sample for an instructor."""

    instructor: str
    sample_path: Path
    created_at: str
    duration_seconds: float

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict."""
        d = asdict(self)
        d["sample_path"] = str(self.sample_path)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> VoiceProfile:
        """Deserialise from a dict (e.g. loaded from JSON)."""
        return cls(
            instructor=data["instructor"],
            sample_path=Path(data["sample_path"]),
            created_at=data["created_at"],
            duration_seconds=float(data["duration_seconds"]),
        )


class VoiceProfileManager:
    """Manage voice reference samples per instructor for consistent TTS dubbing.

    Profiles are stored as WAV files under ``PROFILES_DIR`` with a JSON
    registry that maps instructor slugs to their metadata.
    """

    PROFILES_DIR = Path(__file__).parent / "samples"
    REGISTRY_FILE = Path(__file__).parent / "registry.json"

    def __init__(self) -> None:
        self.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, dict] = self._load_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_sample(
        self,
        video_path: Path,
        instructor: str,
        start_sec: float = 60,
        duration: float = 15,
    ) -> VoiceProfile:
        """Extract a voice sample from a video for an instructor.

        Uses ffmpeg to pull a clean mono 16 kHz WAV segment.

        Parameters
        ----------
        video_path:
            Source video file.
        instructor:
            Instructor name (will be slugified for the filename).
        start_sec:
            Start offset in seconds.
        duration:
            Duration of the sample in seconds.

        Returns
        -------
        The created (or updated) VoiceProfile.

        Raises
        ------
        FileNotFoundError
            If the video does not exist.
        RuntimeError
            If ffmpeg fails.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        slug = self._slugify(instructor)
        output_wav = self.PROFILES_DIR / f"{slug}.wav"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-t", str(duration),
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_wav),
        ]
        log.info("Extracting voice sample: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (rc={result.returncode}): {result.stderr}")

        # Verify the file was created and get its actual duration
        if not output_wav.exists():
            raise RuntimeError("ffmpeg produced no output file")

        actual_duration = self._probe_duration(output_wav)
        if actual_duration <= 0:
            actual_duration = duration

        profile = VoiceProfile(
            instructor=instructor,
            sample_path=output_wav,
            created_at=datetime.now().isoformat(),
            duration_seconds=actual_duration,
        )

        self._registry[slug] = profile.to_dict()
        self._save_registry()
        log.info("Saved voice profile for '%s' (%s, %.1fs)", instructor, output_wav, actual_duration)
        return profile

    def get_profile(self, instructor: str) -> Optional[VoiceProfile]:
        """Get the voice profile for an instructor, or None if not registered."""
        slug = self._slugify(instructor)
        data = self._registry.get(slug)
        if data is None:
            return None
        profile = VoiceProfile.from_dict(data)
        if not profile.sample_path.exists():
            log.warning("Sample file missing for '%s': %s", instructor, profile.sample_path)
            return None
        return profile

    def list_profiles(self) -> list[VoiceProfile]:
        """List all registered voice profiles."""
        profiles: list[VoiceProfile] = []
        for data in self._registry.values():
            try:
                p = VoiceProfile.from_dict(data)
                profiles.append(p)
            except (KeyError, TypeError) as exc:
                log.warning("Skipping malformed registry entry: %s", exc)
        return profiles

    def delete_profile(self, instructor: str) -> bool:
        """Delete a voice profile and its WAV file.

        Returns True if deleted, False if not found.
        """
        slug = self._slugify(instructor)
        data = self._registry.pop(slug, None)
        if data is None:
            log.info("No profile found for '%s'", instructor)
            return False

        sample_path = Path(data.get("sample_path", ""))
        if sample_path.exists():
            sample_path.unlink()
            log.info("Deleted sample file: %s", sample_path)

        self._save_registry()
        log.info("Deleted voice profile for '%s'", instructor)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a name to a filesystem-safe slug."""
        slug = name.lower().strip()
        slug = re.sub(r"[^a-z0-9]+", "_", slug)
        slug = slug.strip("_")
        return slug or "unknown"

    @staticmethod
    def _probe_duration(wav_path: Path) -> float:
        """Use ffprobe to get the duration of a WAV file."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(wav_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as exc:
            log.warning("ffprobe duration failed: %s", exc)
        return 0.0

    def _load_registry(self) -> dict[str, dict]:
        """Load the registry JSON file."""
        if self.REGISTRY_FILE.exists():
            try:
                data = json.loads(self.REGISTRY_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Failed to load registry: %s", exc)
        return {}

    def _save_registry(self) -> None:
        """Persist the registry to disk."""
        try:
            self.REGISTRY_FILE.write_text(
                json.dumps(self._registry, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            log.error("Failed to save registry: %s", exc)
