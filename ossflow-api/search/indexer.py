"""Index all SRT files for cross-instructional full-text search."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

SRT_PATTERN = re.compile(
    r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n$|\Z)",
    re.DOTALL,
)


@dataclass
class SearchResult:
    """A single search hit from the subtitle index."""

    instructional: str
    video_filename: str
    srt_path: str
    subtitle_index: int
    start_time: str
    end_time: str
    text: str
    score: float  # relevance score

    def to_dict(self) -> dict:
        return asdict(self)


class SubtitleIndexer:
    """Index all SRT files for cross-instructional full-text search.

    The index is stored as a single JSON file for simplicity.
    """

    INDEX_PATH = Path(__file__).parent / "index.json"

    def __init__(self) -> None:
        self._entries: list[dict] = []
        self._metadata: dict = {}
        self._load_index()

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def build_index(self, root_dir: Path) -> int:
        """Scan all SRT files under *root_dir*, parse them, and build a search index.

        Returns the number of subtitle entries indexed.
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        entries: list[dict] = []
        file_count = 0

        for dirpath, _dirnames, filenames in os.walk(root_dir):
            dp = Path(dirpath)
            for fname in sorted(filenames):
                if not fname.lower().endswith(".srt"):
                    continue
                srt_path = dp / fname
                instructional = self._guess_instructional(srt_path, root_dir)
                video_filename = srt_path.stem  # SRT typically matches video stem

                try:
                    content = srt_path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    log.warning("Cannot read %s: %s", srt_path, exc)
                    continue

                for m in SRT_PATTERN.finditer(content):
                    entries.append({
                        "path": str(srt_path),
                        "instructional": instructional,
                        "video": video_filename,
                        "index": int(m.group(1)),
                        "start": m.group(2),
                        "end": m.group(3),
                        "text": m.group(4).strip(),
                    })

                file_count += 1

        self._entries = entries
        self._metadata = {
            "total_entries": len(entries),
            "total_files": file_count,
            "root_dir": str(root_dir),
            "last_built": datetime.now().isoformat(),
        }
        self._save_index()
        log.info("Built search index: %d entries from %d SRT files", len(entries), file_count)
        return len(entries)

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 50) -> list[SearchResult]:
        """Search the index for a query string.

        Case-insensitive substring match. Results are sorted by relevance:
        - exact match (full text equals query) -> score 1.0
        - starts with query                    -> score 0.8
        - contains query                       -> score 0.5

        Parameters
        ----------
        query:
            The search string.
        limit:
            Maximum number of results to return.

        Returns
        -------
        List of SearchResult objects sorted by score descending.
        """
        if not query:
            return []

        q_lower = query.lower().strip()
        results: list[SearchResult] = []

        for entry in self._entries:
            text_lower = entry["text"].lower()

            if q_lower not in text_lower:
                continue

            # Score by match quality
            if text_lower == q_lower:
                score = 1.0
            elif text_lower.startswith(q_lower):
                score = 0.8
            else:
                # Boost if the query appears as whole words
                if re.search(r"\b" + re.escape(q_lower) + r"\b", text_lower):
                    score = 0.6
                else:
                    score = 0.4

            results.append(SearchResult(
                instructional=entry["instructional"],
                video_filename=entry["video"],
                srt_path=entry["path"],
                subtitle_index=entry["index"],
                start_time=entry["start"],
                end_time=entry["end"],
                text=entry["text"],
                score=score,
            ))

        # Sort by score descending, then by instructional/index for stability
        results.sort(key=lambda r: (-r.score, r.instructional, r.subtitle_index))
        return results[:limit]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return index statistics."""
        return {
            "total_entries": self._metadata.get("total_entries", len(self._entries)),
            "total_files": self._metadata.get("total_files", 0),
            "root_dir": self._metadata.get("root_dir", ""),
            "last_built": self._metadata.get("last_built", ""),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Load a previously built index from disk."""
        if not self.INDEX_PATH.exists():
            return
        try:
            raw = json.loads(self.INDEX_PATH.read_text(encoding="utf-8"))
            self._entries = raw.get("entries", [])
            self._metadata = raw.get("metadata", {})
            log.info("Loaded search index: %d entries", len(self._entries))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load search index: %s", exc)

    def _save_index(self) -> None:
        """Persist the index to disk."""
        payload = {
            "metadata": self._metadata,
            "entries": self._entries,
        }
        try:
            self.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.INDEX_PATH.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            log.error("Failed to save search index: %s", exc)

    @staticmethod
    def _guess_instructional(srt_path: Path, root_dir: Path) -> str:
        """Guess the instructional name from the file's relative path.

        Heuristic: use the first directory component under root_dir.
        """
        try:
            rel = srt_path.relative_to(root_dir)
            parts = rel.parts
            if len(parts) > 1:
                return parts[0]
            return srt_path.stem
        except ValueError:
            return srt_path.stem
