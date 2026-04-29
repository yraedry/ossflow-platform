"""Generate and embed MKV chapter metadata using the Matroska XML format."""

from __future__ import annotations

import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.dom import minidom

log = logging.getLogger(__name__)


class MkvChapterGenerator:
    """Generate and embed MKV chapter metadata for native player navigation."""

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seconds_to_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS.nnnnnnnnn Matroska timestamp."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:012.9f}"

    @staticmethod
    def _timestamp_to_seconds(ts: str) -> float:
        """Convert HH:MM:SS.nnn... to seconds."""
        parts = ts.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])

    # ------------------------------------------------------------------
    # XML generation
    # ------------------------------------------------------------------

    def generate_xml(self, chapters: list[dict[str, Any]], output_path: Path) -> Path:
        """Generate a Matroska chapter XML file from a list of chapter dicts.

        Parameters
        ----------
        chapters:
            Each dict must contain ``title`` (str), ``start`` (float, seconds)
            and ``end`` (float, seconds).
        output_path:
            Destination path for the XML file.

        Returns
        -------
        Path to the written XML file.
        """
        if not chapters:
            raise ValueError("chapters list must not be empty")

        root = ET.Element("Chapters")
        edition = ET.SubElement(root, "EditionEntry")

        uid_counter = 1
        for ch in chapters:
            atom = ET.SubElement(edition, "ChapterAtom")

            uid_el = ET.SubElement(atom, "ChapterUID")
            uid_el.text = str(uid_counter)
            uid_counter += 1

            start_el = ET.SubElement(atom, "ChapterTimeStart")
            start_el.text = self._seconds_to_timestamp(float(ch["start"]))

            end_el = ET.SubElement(atom, "ChapterTimeEnd")
            end_el.text = self._seconds_to_timestamp(float(ch["end"]))

            display = ET.SubElement(atom, "ChapterDisplay")

            title_el = ET.SubElement(display, "ChapterString")
            title_el.text = str(ch.get("title", f"Chapter {uid_counter - 1}"))

            lang_el = ET.SubElement(display, "ChapterLanguage")
            lang_el.text = "eng"

        # Pretty-print
        rough = ET.tostring(root, encoding="unicode")
        parsed = minidom.parseString(rough)
        pretty = parsed.toprettyxml(indent="  ", encoding="utf-8")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(pretty)
        log.info("Wrote chapter XML to %s (%d chapters)", output_path, len(chapters))
        return output_path

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    def embed_chapters(self, video_path: Path, chapters_xml: Path) -> bool:
        """Embed chapters into an MKV file using mkvmerge.

        Creates a new file alongside the original with ``_chaptered`` suffix,
        then replaces the original.

        Parameters
        ----------
        video_path:
            Path to the source MKV file.
        chapters_xml:
            Path to the Matroska chapter XML file.

        Returns
        -------
        True on success, False on failure.
        """
        video_path = Path(video_path)
        chapters_xml = Path(chapters_xml)

        if not video_path.exists():
            log.error("Video file not found: %s", video_path)
            return False
        if not chapters_xml.exists():
            log.error("Chapters XML not found: %s", chapters_xml)
            return False

        output = video_path.with_name(video_path.stem + "_chaptered" + video_path.suffix)

        cmd = [
            "mkvmerge",
            "-o", str(output),
            "--chapters", str(chapters_xml),
            str(video_path),
        ]
        log.info("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode not in (0, 1):  # mkvmerge returns 1 for warnings
                log.error("mkvmerge failed (rc=%d): %s", result.returncode, result.stderr)
                return False

            # Replace original with chaptered version
            video_path.unlink()
            output.rename(video_path)
            log.info("Embedded chapters into %s", video_path)
            return True

        except FileNotFoundError:
            log.error("mkvmerge not found on PATH. Install MKVToolNix.")
            return False
        except subprocess.TimeoutExpired:
            log.error("mkvmerge timed out for %s", video_path)
            return False
        except Exception as exc:
            log.error("Failed to embed chapters: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def extract_chapters(self, video_path: Path) -> list[dict[str, Any]]:
        """Extract existing chapters from an MKV file using mkvextract.

        Returns
        -------
        List of dicts with ``title``, ``start`` and ``end`` (in seconds).
        Empty list if no chapters found or on error.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            log.error("Video file not found: %s", video_path)
            return []

        tmp_xml = video_path.with_suffix(".chapters.xml")

        cmd = ["mkvextract", str(video_path), "chapters", str(tmp_xml)]
        log.info("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                log.warning("mkvextract returned %d: %s", result.returncode, result.stderr)
                return []

            if not tmp_xml.exists() or tmp_xml.stat().st_size == 0:
                log.info("No chapters found in %s", video_path)
                return []

            chapters = self._parse_chapter_xml(tmp_xml)
            return chapters

        except FileNotFoundError:
            log.error("mkvextract not found on PATH. Install MKVToolNix.")
            return []
        except subprocess.TimeoutExpired:
            log.error("mkvextract timed out for %s", video_path)
            return []
        except Exception as exc:
            log.error("Failed to extract chapters: %s", exc)
            return []
        finally:
            if tmp_xml.exists():
                tmp_xml.unlink()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_chapter_xml(self, xml_path: Path) -> list[dict[str, Any]]:
        """Parse a Matroska chapter XML file into a list of chapter dicts."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        chapters: list[dict[str, Any]] = []

        for atom in root.iter("ChapterAtom"):
            start_el = atom.find("ChapterTimeStart")
            end_el = atom.find("ChapterTimeEnd")
            display = atom.find("ChapterDisplay")
            title_el = display.find("ChapterString") if display is not None else None

            start = self._timestamp_to_seconds(start_el.text) if start_el is not None and start_el.text else 0.0
            end = self._timestamp_to_seconds(end_el.text) if end_el is not None and end_el.text else 0.0
            title = title_el.text if title_el is not None and title_el.text else "Untitled"

            chapters.append({"title": title, "start": start, "end": end})

        log.info("Parsed %d chapters from %s", len(chapters), xml_path)
        return chapters
