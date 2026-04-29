"""Utility functions for filename sanitization, season extraction, and logging."""

import logging
import re
import sys

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FMT = "%H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FMT, datefmt=LOG_DATE_FMT,
                        stream=sys.stdout)


def sanitize_filename(text: str) -> str:
    """Clean a string for use as a filename component."""
    t = re.sub(r'[\\/*?:"<>|^~`\xa9\xae;{}]', "", text)
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r" +", " ", t).strip()
    return t


def extract_season_number(filename: str, fallback: int) -> int:
    """Extract volume/part/disc number from filename, or use fallback."""
    m = re.search(r"(?:vol|volume|part|disc|disk)[ _.-]*(\d+)",
                  filename, re.IGNORECASE)
    return int(m.group(1)) if m else fallback
