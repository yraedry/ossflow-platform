"""Weighted background text memory with decay, timestamps, and garbage collection."""

import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from ..config import Config


@dataclass
class MemoryEntry:
    """A single background text entry with weight and timestamp."""
    text: str
    weight: float
    last_seen: float = field(default_factory=time.monotonic)


class BackgroundMemory:
    """Weighted background text memory with decay, timestamped entries,
    and garbage collection.

    Improvements over v1:
    - Entries have timestamps for temporal expiration
    - Garbage collection removes entries with weight < 0.1
    - Weight is used in is_background: entries with weight > 5.0 are
      considered definitive background
    - Can learn during both voice and detection phases
    """

    # Weight above which an entry is considered definitive background
    DEFINITIVE_WEIGHT: float = 5.0
    # Weight below which entries are garbage-collected
    GC_WEIGHT_THRESHOLD: float = 0.1

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self._entries: list[MemoryEntry] = []

    @property
    def entries_as_tuples(self) -> list[tuple[str, float]]:
        """Legacy compatibility: return entries as (text, weight) tuples."""
        return [(e.text, e.weight) for e in self._entries]

    def learn(self, text: str) -> None:
        """Add or boost a background text entry."""
        text_upper = text.upper().strip()
        if len(text_upper) <= 3:
            return

        now = time.monotonic()

        # Check for existing similar entry
        for entry in self._entries:
            ratio = SequenceMatcher(None, entry.text, text_upper).ratio()
            if ratio > self.cfg.background_boost_threshold:
                # Boost existing entry
                entry.weight += 1.0
                entry.last_seen = now
                return

        # New entry
        self._entries.append(MemoryEntry(
            text=text_upper,
            weight=1.0,
            last_seen=now,
        ))

        # Prune by weight if over capacity
        if len(self._entries) > self.cfg.background_max_entries:
            self._entries.sort(key=lambda e: e.weight)
            self._entries = self._entries[
                len(self._entries) - self.cfg.background_max_entries:
            ]

    def decay(self) -> None:
        """Apply decay to all weights and run garbage collection."""
        factor = self.cfg.background_decay
        self._entries = [
            MemoryEntry(text=e.text, weight=e.weight * factor, last_seen=e.last_seen)
            for e in self._entries
        ]
        self._garbage_collect()

    def _garbage_collect(self) -> None:
        """Remove entries with weight below the GC threshold."""
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if e.weight >= self.GC_WEIGHT_THRESHOLD
        ]
        removed = before - len(self._entries)
        if removed > 0:
            import logging
            logging.getLogger(__name__).debug(
                "GC removed %d low-weight entries", removed)

    def is_background(self, text: str) -> bool:
        """Return True if text matches a known background entry.

        Entries with weight > DEFINITIVE_WEIGHT are considered definitive
        background regardless of similarity nuances.
        """
        text_upper = text.upper().strip()
        if not self._entries:
            return False

        for entry in self._entries:
            ratio = SequenceMatcher(None, entry.text, text_upper).ratio()
            if ratio > self.cfg.background_similarity:
                # High-weight entries are definitive background
                if entry.weight > self.DEFINITIVE_WEIGHT:
                    return True
                # Normal similarity match
                return True

        return False
