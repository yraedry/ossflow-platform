"""Data containers for OCR results and detected chapters."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OcrResult:
    text: str
    confidence: float
    raw_texts: list[str] = field(default_factory=list)


@dataclass
class Chapter:
    start: float
    end: Optional[float]
    title: str
