"""CLI eval: verifies the BJJFanatics oracle flow end-to-end on a known product.

Usage (inside the `chapter-splitter` container):

    python -m scripts.eval_oracle \
        --title "Tripod Passing" \
        --author "Jozef Chen" \
        --instructional-dir "/media/instruccionales/Tripod Passing - Jozef Chen" \
        [--dry-run]

Exit codes:
    0 - all checks PASS
    1 - one or more checks FAIL
    2 - no search candidate above min_score threshold
    3 - oracle structural checks failed (wrong volumes/chapters/timestamps)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("eval_oracle")

MIN_CANDIDATE_SCORE = 0.7


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def render(self) -> str:
        badge = "PASS" if self.ok else "FAIL"
        suffix = f" - {self.detail}" if self.detail else ""
        return f"  [{badge}] {self.name}{suffix}"


def _fmt_ts(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval_oracle",
        description="Verify BJJFanatics oracle flow on Tripod Passing - Jozef Chen.",
    )
    p.add_argument("--title", required=True)
    p.add_argument("--author", required=True)
    p.add_argument(
        "--instructional-dir",
        required=True,
        help="Root dir of the instructional (mp4 files expected inside).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not run OracleSplitter.split(); only verify oracle data.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args()


def _run_search(provider, title: str, author: str, checks: list[Check]) -> Optional[object]:
    from chapter_splitter.oracle import OracleError

    try:
        candidates = provider.search(title, author)
    except OracleError as e:
        checks.append(Check("search() returns candidates", False, f"{type(e).__name__}: {e}"))
        return None

    print("\nTop-3 search candidates:")
    for c in candidates[:3]:
        print(f"  score={c.score:.3f}  {c.title!r}  {c.url}")

    if not candidates:
        checks.append(Check("search() returns candidates", False, "empty result"))
        return None

    top = candidates[0]
    ok = top.score > MIN_CANDIDATE_SCORE
    checks.append(
        Check(
            f"top candidate score > {MIN_CANDIDATE_SCORE}",
            ok,
            f"got {top.score:.3f} ({top.title!r})",
        )
    )
    return top if ok else None


def _verify_oracle(oracle, checks: list[Check]) -> bool:
    """Apply all Tripod Passing structural expectations. Returns True if all OK."""
    all_ok = True

    n_vols = len(oracle.volumes)
    print(f"\nOracle summary: {n_vols} volumes, provider={oracle.provider_id}")
    for v in oracle.volumes:
        print(
            f"  Volume {v.number}: {len(v.chapters)} chapters, "
            f"total={_fmt_ts(v.total_duration_s)} ({v.total_duration_s:.1f}s)"
        )

    c = Check(">= 6 volumes detected", n_vols >= 6, f"got {n_vols}")
    checks.append(c)
    all_ok &= c.ok

    vol1 = oracle.volume(1)
    if vol1 is None:
        checks.append(Check("Volume 1 present", False, "missing"))
        return False
    checks.append(Check("Volume 1 present", True))

    c = Check("Volume 1 has 12 chapters", len(vol1.chapters) == 12, f"got {len(vol1.chapters)}")
    checks.append(c)
    all_ok &= c.ok

    if len(vol1.chapters) >= 1:
        ch1 = vol1.chapters[0]
        c = Check(
            "V1.ch1 title == 'Phases Of Engagement'",
            ch1.title == "Phases Of Engagement",
            f"got {ch1.title!r}",
        )
        checks.append(c); all_ok &= c.ok
        c = Check("V1.ch1 start_s == 0", ch1.start_s == 0, f"got {ch1.start_s}")
        checks.append(c); all_ok &= c.ok

    if len(vol1.chapters) >= 2:
        ch2 = vol1.chapters[1]
        c = Check(
            "V1.ch2 title == 'Prerequisites To Pass And How The Tripod Fits In'",
            ch2.title == "Prerequisites To Pass And How The Tripod Fits In",
            f"got {ch2.title!r}",
        )
        checks.append(c); all_ok &= c.ok
        c = Check("V1.ch2 start_s == 95 (1:35)", ch2.start_s == 95, f"got {ch2.start_s}")
        checks.append(c); all_ok &= c.ok

    # Per-volume duration + monotonic starts.
    dur_ok = True
    mono_ok = True
    for v in oracle.volumes:
        if v.total_duration_s <= 0:
            dur_ok = False
            logger.error("Volume %d has total_duration_s <= 0", v.number)
        prev = -1.0
        for ch in v.chapters:
            if ch.start_s < prev:
                mono_ok = False
                logger.error("Volume %d non-monotonic start_s", v.number)
                break
            prev = ch.start_s
    c = Check("All volumes total_duration_s > 0", dur_ok)
    checks.append(c); all_ok &= c.ok
    c = Check("Chapter starts are monotonic in every volume", mono_ok)
    checks.append(c); all_ok &= c.ok

    return all_ok


def _persist_oracle(instructional_dir: Path, oracle) -> None:
    meta_file = instructional_dir / ".bjj-meta.json"
    data: dict = {}
    if meta_file.exists():
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not parse existing %s, overwriting", meta_file)
            data = {}
    data["oracle"] = json.loads(oracle.model_dump_json())
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Persisted oracle to %s", meta_file)


def _run_split(instructional_dir: Path, oracle, checks: list[Check]) -> None:
    from chapter_splitter.splitting.oracle_splitter import OracleSplitter

    splitter = OracleSplitter(instructional_dir=instructional_dir, oracle=oracle)

    def _progress(pct: float, msg: str) -> None:
        print(f"  [{pct:5.1f}%] {msg}")

    report = splitter.split(progress_cb=_progress)
    print(f"\nSplit report: {report.to_dict()}")

    season1 = instructional_dir / "Season 01"
    if not season1.exists():
        checks.append(Check("Season 01/ created", False, f"missing {season1}"))
        return
    files = sorted(p.name for p in season1.iterdir() if p.is_file())
    c = Check(
        "Season 01/ contains 12 files",
        len(files) == 12,
        f"got {len(files)}",
    )
    checks.append(c)
    for f in files:
        print(f"    - {f}")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    from chapter_splitter.oracle import OracleError, discover, registry

    discover()
    try:
        provider = registry.get("bjjfanatics")
    except OracleError as e:
        print(f"ERROR resolving provider: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    checks: list[Check] = []

    top = _run_search(provider, args.title, args.author, checks)
    if top is None:
        _print_summary(checks)
        return 2

    try:
        oracle = provider.scrape(top.url)
    except OracleError as e:
        print(f"ERROR scraping {top.url}: {type(e).__name__}: {e}", file=sys.stderr)
        checks.append(Check("scrape() succeeds", False, f"{type(e).__name__}: {e}"))
        _print_summary(checks)
        return 1
    checks.append(Check("scrape() succeeds", True))

    oracle_ok = _verify_oracle(oracle, checks)
    if not oracle_ok:
        _print_summary(checks)
        return 3

    instructional_dir = Path(args.instructional_dir)
    # Persist oracle regardless of dry-run so the frontend can read it.
    if instructional_dir.exists():
        try:
            _persist_oracle(instructional_dir, oracle)
            checks.append(Check(".bjj-meta.json updated with oracle", True))
        except Exception as e:
            checks.append(Check(".bjj-meta.json updated with oracle", False, str(e)))
    else:
        checks.append(
            Check(
                ".bjj-meta.json updated with oracle",
                False,
                f"instructional dir missing: {instructional_dir}",
            )
        )

    if not args.dry_run:
        if not instructional_dir.exists():
            checks.append(
                Check("Splitter run", False, f"dir missing: {instructional_dir}")
            )
        else:
            try:
                _run_split(instructional_dir, oracle, checks)
            except Exception as e:
                logger.exception("split failed")
                checks.append(Check("Splitter run", False, f"{type(e).__name__}: {e}"))
    else:
        print("\n(dry-run: skipping OracleSplitter.split())")

    return _print_summary(checks)


def _print_summary(checks: list[Check]) -> int:
    print("\n==================== SUMMARY ====================")
    for c in checks:
        print(c.render())
    failed = [c for c in checks if not c.ok]
    print("-------------------------------------------------")
    print(f"  {len(checks) - len(failed)}/{len(checks)} PASS")
    print("=================================================")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
