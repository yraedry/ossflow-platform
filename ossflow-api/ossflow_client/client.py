"""Client to export processed data to the OssFlow Spring Boot backend."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SRT_PATTERN = re.compile(
    r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n$|\Z)",
    re.DOTALL,
)


@dataclass
class OssFlowConfig:
    """Connection settings for the OssFlow backend."""

    base_url: str = "http://localhost:8080"
    api_prefix: str = "/api/v1"


class OssFlowClient:
    """Client to export processed data to the OssFlow Spring Boot backend.

    Uses the ``requests`` library to communicate with the REST API.
    """

    def __init__(self, config: OssFlowConfig | None = None) -> None:
        self.config = config or OssFlowConfig()
        self._base = f"{self.config.base_url}{self.config.api_prefix}"

    # ------------------------------------------------------------------
    # Core API methods
    # ------------------------------------------------------------------

    def create_instructional(self, name: str, instructor: str, path: str) -> dict:
        """Create an instructional record.

        POST /api/v1/instructionals
        """
        import requests

        payload = {"name": name, "instructor": instructor, "path": path}
        log.info("POST %s/instructionals -> %s", self._base, payload)
        resp = requests.post(f"{self._base}/instructionals", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def import_chapters(self, instructional_id: int, chapters: list[dict]) -> dict:
        """Import chapter metadata for an instructional.

        POST /api/v1/instructionals/{id}/import-chapters

        Parameters
        ----------
        instructional_id:
            The ID of the instructional.
        chapters:
            List of dicts with ``title``, ``start`` (seconds), ``end`` (seconds).
        """
        import requests

        url = f"{self._base}/instructionals/{instructional_id}/import-chapters"
        log.info("POST %s (%d chapters)", url, len(chapters))
        resp = requests.post(url, json={"chapters": chapters}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def import_subtitles(self, chapter_id: int, srt_path: Path) -> dict:
        """Parse an SRT file and POST subtitles to OssFlow.

        POST /api/v1/chapters/{chapter_id}/import-subtitles

        Parameters
        ----------
        chapter_id:
            The chapter ID in OssFlow.
        srt_path:
            Path to the SRT file to parse and upload.

        Returns
        -------
        Response from the backend with import summary.
        """
        import requests

        srt_path = Path(srt_path)
        if not srt_path.exists():
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        content = srt_path.read_text(encoding="utf-8", errors="replace")
        subtitles: list[dict] = []

        for m in SRT_PATTERN.finditer(content):
            subtitles.append({
                "index": int(m.group(1)),
                "startTime": m.group(2),
                "endTime": m.group(3),
                "text": m.group(4).strip(),
            })

        if not subtitles:
            log.warning("No subtitles found in %s", srt_path)
            return {"imported": 0}

        url = f"{self._base}/chapters/{chapter_id}/import-subtitles"
        log.info("POST %s (%d subtitles from %s)", url, len(subtitles), srt_path.name)
        resp = requests.post(url, json={"subtitles": subtitles}, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def create_study_note(
        self,
        chapter_id: int,
        timestamp: float,
        content: str,
        tags: str = "",
    ) -> dict:
        """Create a study note for a chapter.

        POST /api/v1/study-notes
        """
        import requests

        payload = {
            "chapterId": chapter_id,
            "timestamp": timestamp,
            "content": content,
            "tags": tags,
        }
        log.info("POST %s/study-notes for chapter %d", self._base, chapter_id)
        resp = requests.post(f"{self._base}/study-notes", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str) -> list[dict]:
        """Search across all indexed content in OssFlow.

        GET /api/v1/search?q=query
        """
        import requests

        log.info("GET %s/search?q=%s", self._base, query)
        resp = requests.get(f"{self._base}/search", params={"q": query}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> bool:
        """Check if the OssFlow backend is reachable.

        Returns True if the backend responds, False otherwise.
        """
        import requests

        try:
            resp = requests.get(
                f"{self.config.base_url}/actuator/health",
                timeout=5,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    # High-level pipeline
    # ------------------------------------------------------------------

    def export_full_instructional(self, root_dir: Path, instructor: str) -> dict:
        """Full pipeline: create instructional, scan chapters, import subtitles.

        Parameters
        ----------
        root_dir:
            Directory containing the instructional's video and SRT files.
        instructor:
            Name of the instructor.

        Returns
        -------
        Summary dict with keys: instructional_id, chapters_imported,
        subtitles_imported, errors.
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        instr_name = root_dir.name
        summary: dict = {
            "instructional_id": None,
            "chapters_imported": 0,
            "subtitles_imported": 0,
            "errors": [],
        }

        # Step 1: create instructional
        try:
            result = self.create_instructional(instr_name, instructor, str(root_dir))
            instr_id = result.get("id")
            summary["instructional_id"] = instr_id
            log.info("Created instructional %s (id=%s)", instr_name, instr_id)
        except Exception as exc:
            log.error("Failed to create instructional: %s", exc)
            summary["errors"].append(f"create_instructional: {exc}")
            return summary

        # Step 2: scan for video files and build chapter list
        video_extensions = {".mp4", ".mkv", ".avi", ".mov"}
        chapters: list[dict] = []
        video_srt_map: dict[str, Path] = {}  # chapter title -> srt path

        for dirpath, _dirnames, filenames in os.walk(root_dir):
            dp = Path(dirpath)
            for fname in sorted(filenames):
                fp = dp / fname
                if fp.suffix.lower() in video_extensions:
                    # Use the filename stem as chapter title
                    title = fp.stem
                    chapters.append({
                        "title": title,
                        "start": 0.0,
                        "end": 0.0,
                        "file": str(fp),
                    })
                    # Look for matching SRT
                    srt_path = fp.with_suffix(".srt")
                    if srt_path.exists():
                        video_srt_map[title] = srt_path

        if chapters:
            try:
                ch_result = self.import_chapters(instr_id, chapters)
                imported_chapters = ch_result.get("chapters", [])
                summary["chapters_imported"] = len(imported_chapters)
                log.info("Imported %d chapters", len(imported_chapters))

                # Step 3: import subtitles for each chapter
                for imported_ch in imported_chapters:
                    ch_title = imported_ch.get("title", "")
                    ch_id = imported_ch.get("id")
                    srt_path = video_srt_map.get(ch_title)

                    if srt_path and ch_id:
                        try:
                            sub_result = self.import_subtitles(ch_id, srt_path)
                            count = sub_result.get("imported", 0)
                            summary["subtitles_imported"] += count
                            log.info("Imported %d subtitles for '%s'", count, ch_title)
                        except Exception as exc:
                            log.error("Failed to import subtitles for '%s': %s", ch_title, exc)
                            summary["errors"].append(f"import_subtitles({ch_title}): {exc}")

            except Exception as exc:
                log.error("Failed to import chapters: %s", exc)
                summary["errors"].append(f"import_chapters: {exc}")

        log.info("Export complete: %s", summary)
        return summary
