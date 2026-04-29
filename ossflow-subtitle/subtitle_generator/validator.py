"""Final validation pass on generated SRT subtitle files."""

from __future__ import annotations

import logging
from typing import Optional

from .config import SubtitleConfig
from .utils import format_timestamp

log = logging.getLogger("subtitler")


class SubtitleValidator:
    """Final validation pass on generated SRT files."""

    def __init__(self, config: SubtitleConfig) -> None:
        self.config = config

    def validate(self, subtitles: list[dict], audio_duration: Optional[float] = None) -> dict:
        """Validate subtitles and return a summary report dict."""
        issues: list[str] = []
        gaps_over_threshold: list[dict] = []

        # Check each subtitle
        for i, sub in enumerate(subtitles):
            idx = i + 1
            duration = sub["end"] - sub["start"]
            text = sub.get("text", "")

            if not text.strip():
                issues.append(f"Subtitle {idx}: empty text")

            if duration < self.config.min_duration - 0.01:
                issues.append(f"Subtitle {idx}: duration {duration:.2f}s below minimum {self.config.min_duration}s")

            if i > 0:
                gap = sub["start"] - subtitles[i - 1]["end"]
                if gap < -0.01:
                    issues.append(f"Subtitle {idx}: overlaps with previous by {abs(gap):.3f}s")
                if gap > self.config.gap_warn_threshold:
                    gaps_over_threshold.append({
                        "after_subtitle": idx - 1,
                        "gap_seconds": gap,
                        "at_time": format_timestamp(subtitles[i - 1]["end"]),
                    })

        # Coverage calculation
        total_subtitle_time = sum(sub["end"] - sub["start"] for sub in subtitles)
        if subtitles:
            span = subtitles[-1]["end"] - subtitles[0]["start"]
        else:
            span = 0
        coverage = (total_subtitle_time / span * 100) if span > 0 else 0

        report = {
            "total_segments": len(subtitles),
            "total_subtitle_time": total_subtitle_time,
            "span": span,
            "coverage_percent": coverage,
            "issues": issues,
            "large_gaps": gaps_over_threshold,
        }

        # Log summary
        log.info(
            "Validation: %d segments, coverage %.1f%%, %d issues, %d large gaps (>%.0fs)",
            report["total_segments"],
            report["coverage_percent"],
            len(report["issues"]),
            len(report["large_gaps"]),
            self.config.gap_warn_threshold,
        )
        for issue in issues:
            log.warning("  %s", issue)
        for gap in gaps_over_threshold:
            log.warning(
                "  Gap of %.1fs after subtitle %d (at %s)",
                gap["gap_seconds"], gap["after_subtitle"], gap["at_time"],
            )

        return report
