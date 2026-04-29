"""Post-generation SRT quality analysis: flag likely hallucinations."""

from __future__ import annotations

from difflib import SequenceMatcher

from .config import SubtitleConfig


def _severity_rank(sev: str) -> int:
    return {"error": 3, "warn": 2, "info": 1}.get(sev, 0)


class SubtitleQualityChecker:
    """Analyse generated SRT subtitles and report per-segment issues."""

    def __init__(self, config: SubtitleConfig | None = None) -> None:
        self.config = config or SubtitleConfig()

    def check(self, subs: list[dict]) -> dict:
        """Return {segments: [{idx, start, end, text, issues, severity}], summary}."""
        segments: list[dict] = []
        total_dur = 0.0
        for i, sub in enumerate(subs):
            issues = self._issues_for(i, sub, subs)
            severity = "ok"
            for it in issues:
                if _severity_rank(it["severity"]) > _severity_rank(severity):
                    severity = it["severity"]
            dur = max(0.0, sub["end"] - sub["start"])
            total_dur += dur
            segments.append({
                "idx": i + 1,
                "start": sub["start"],
                "end": sub["end"],
                "text": sub.get("text", ""),
                "duration": dur,
                "issues": issues,
                "severity": severity,
            })

        span = (subs[-1]["end"] - subs[0]["start"]) if subs else 0.0
        coverage = (total_dur / span * 100) if span > 0 else 0.0
        flagged = [s for s in segments if s["severity"] != "ok"]
        counts = {"error": 0, "warn": 0, "info": 0}
        for s in flagged:
            counts[s["severity"]] = counts.get(s["severity"], 0) + 1

        return {
            "segments": segments,
            "summary": {
                "total": len(segments),
                "flagged": len(flagged),
                "by_severity": counts,
                "coverage_percent": coverage,
                "span_seconds": span,
            },
        }

    def _issues_for(self, i: int, sub: dict, subs: list[dict]) -> list[dict]:
        cfg = self.config
        issues: list[dict] = []
        text = (sub.get("text") or "").strip()
        start = sub.get("start", 0.0)
        end = sub.get("end", 0.0)
        duration = end - start

        if not text:
            issues.append({"code": "empty", "severity": "error",
                           "message": "Segmento sin texto"})
        if duration <= 0:
            issues.append({"code": "impossible_timing", "severity": "error",
                           "message": f"Duración inválida ({duration:.2f}s)"})
        elif duration < cfg.min_duration - 0.01:
            issues.append({"code": "too_short", "severity": "warn",
                           "message": f"Duración {duration:.2f}s < mínimo {cfg.min_duration}s"})
        elif duration > cfg.max_duration + 0.01:
            issues.append({"code": "too_long", "severity": "warn",
                           "message": f"Duración {duration:.2f}s > máximo {cfg.max_duration}s"})

        if text and duration > 0:
            cps = len(text.replace("\n", " ")) / duration
            if cps > cfg.max_chars_per_second:
                issues.append({"code": "high_cps", "severity": "warn",
                               "message": f"{cps:.1f} chars/s — probable alucinación"})

        if i > 0:
            prev = subs[i - 1]
            gap = start - prev["end"]
            if gap < -0.01:
                issues.append({"code": "overlap", "severity": "warn",
                               "message": f"Solapa con anterior {abs(gap):.2f}s"})
            if gap > cfg.gap_warn_threshold:
                issues.append({"code": "large_gap", "severity": "info",
                               "message": f"Hueco {gap:.1f}s tras el anterior"})

            prev_text = (prev.get("text") or "").strip().lower()
            cur_text = text.lower()
            if prev_text and cur_text:
                ratio = SequenceMatcher(None, prev_text, cur_text).ratio()
                if ratio >= cfg.similarity_threshold:
                    issues.append({"code": "repeated_segment", "severity": "error",
                                   "message": f"Texto ~{int(ratio*100)}% igual al anterior"})

        if text:
            words = text.replace("\n", " ").split()
            if len(words) >= cfg.repeated_ngram_size * (cfg.repeated_ngram_max + 1):
                seen: dict[str, int] = {}
                n = cfg.repeated_ngram_size
                for k in range(len(words) - n + 1):
                    gram = " ".join(w.lower() for w in words[k:k + n])
                    seen[gram] = seen.get(gram, 0) + 1
                repeated = [g for g, c in seen.items() if c > cfg.repeated_ngram_max]
                if repeated:
                    issues.append({"code": "repeated_phrase", "severity": "error",
                                   "message": f"Frase repetida: {repeated[0][:40]!r}"})

        return issues
