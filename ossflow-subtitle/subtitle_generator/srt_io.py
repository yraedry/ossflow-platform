"""SRT parse + serialize with .bak backup."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .utils import format_timestamp

_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,\.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,\.](\d{3})"
)


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: Path) -> list[dict]:
    """Parse an SRT file into a list of {idx, start, end, text} dicts."""
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\r?\n\r?\n+", raw.strip())
    result: list[dict] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        ts_line_idx = 0
        if lines[0].strip().isdigit():
            ts_line_idx = 1
        m = _TS_RE.search(lines[ts_line_idx])
        if not m:
            continue
        start = _ts_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _ts_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        text_lines = lines[ts_line_idx + 1 :]
        result.append({
            "idx": len(result) + 1,
            "start": start,
            "end": end,
            "text": "\n".join(text_lines).strip(),
        })
    return result


def serialize_srt(subs: list[dict]) -> str:
    """Render subs list back to SRT text."""
    out: list[str] = []
    for i, s in enumerate(subs, 1):
        out.append(str(i))
        out.append(f"{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}")
        out.append(s.get("text", "").strip())
        out.append("")
    return "\n".join(out) + "\n"


def write_srt_with_backup(path: Path, subs: list[dict]) -> Path:
    """Write SRT atomically keeping .srt.bak of previous content."""
    path = Path(path)
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
    path.write_text(serialize_srt(subs), encoding="utf-8")
    return path
