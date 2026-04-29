"""Export processed videos with Plex/Jellyfin-compatible naming and structure."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}


class PlexExporter:
    """Export processed videos with Plex-compatible naming and metadata.

    Plex recognises show-style naming:
        Show Name/Season 01/Show Name - S01E01 - Episode Title.mkv

    This exporter takes a flat or semi-structured source directory and
    reorganises it into that layout.
    """

    def export(
        self,
        instructional_name: str,
        chapters: list[dict],
        source_dir: Path,
        output_dir: Path,
    ) -> None:
        """Create a Plex-compatible folder structure.

        Parameters
        ----------
        instructional_name:
            Human-readable name of the instructional (e.g. "Inside Camping - Gordon Ryan").
        chapters:
            List of dicts, each with at least ``title`` (str) and ``file`` (str,
            filename relative to *source_dir*).  Optionally ``season`` (int, default 1)
            and ``episode`` (int, auto-incremented).
        source_dir:
            Directory containing the source video (and subtitle) files.
        output_dir:
            Root destination directory where the Plex tree will be created.
        """
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)

        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

        safe_name = self._sanitize(instructional_name)
        show_dir = output_dir / safe_name

        episode_counter: dict[int, int] = {}  # season -> next episode number

        for ch in chapters:
            season = int(ch.get("season", 1))
            episode = int(ch.get("episode", 0))
            if episode == 0:
                episode_counter.setdefault(season, 0)
                episode_counter[season] += 1
                episode = episode_counter[season]
            else:
                episode_counter[season] = max(episode_counter.get(season, 0), episode)

            title = ch.get("title", f"Chapter {episode}")
            src_file = ch.get("file", "")
            if not src_file:
                log.warning("Chapter '%s' has no file reference, skipping", title)
                continue

            src_path = source_dir / src_file
            if not src_path.exists():
                log.warning("Source file not found: %s, skipping", src_path)
                continue

            season_dir = show_dir / f"Season {season:02d}"
            season_dir.mkdir(parents=True, exist_ok=True)

            safe_title = self._sanitize(title)
            base_name = f"{safe_name} - S{season:02d}E{episode:02d} - {safe_title}"

            # Copy video
            dst_video = season_dir / f"{base_name}{src_path.suffix}"
            log.info("Copying %s -> %s", src_path, dst_video)
            shutil.copy2(str(src_path), str(dst_video))

            # Copy matching subtitle files
            for ext in SUBTITLE_EXTENSIONS:
                srt_src = src_path.with_suffix(ext)
                if srt_src.exists():
                    dst_srt = season_dir / f"{base_name}{ext}"
                    log.info("Copying subtitle %s -> %s", srt_src, dst_srt)
                    shutil.copy2(str(srt_src), str(dst_srt))

                # Also check language-tagged variants (e.g. .ES.srt)
                for lang_tag in ("ES", "EN", "PT"):
                    lang_src = src_path.with_suffix(f".{lang_tag}{ext}")
                    if lang_src.exists():
                        dst_lang = season_dir / f"{base_name}.{lang_tag}{ext}"
                        log.info("Copying subtitle %s -> %s", lang_src, dst_lang)
                        shutil.copy2(str(lang_src), str(dst_lang))

        total_files = sum(1 for _ in show_dir.rglob("*") if _.is_file()) if show_dir.exists() else 0
        log.info(
            "Plex export complete: %s -> %s (%d files)",
            instructional_name,
            show_dir,
            total_files,
        )

    @staticmethod
    def _sanitize(name: str) -> str:
        """Sanitize a name for filesystem use."""
        cleaned = re.sub(r'[<>:"/\\|?*]', "", name)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
