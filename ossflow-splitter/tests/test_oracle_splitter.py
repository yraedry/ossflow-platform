"""Tests for :class:`chapter_splitter.splitting.oracle_splitter.OracleSplitter`."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from chapter_splitter.oracle.models import (
    OracleChapter,
    OracleResult,
    OracleVolume,
)
from chapter_splitter.splitting.oracle_splitter import (
    OracleSplitter,
    SplitReport,
)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not in PATH",
)


def _make_video(path: Path, duration_s: int) -> None:
    """Generate a black mp4 of given duration via ffmpeg lavfi."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"color=c=black:s=160x120:r=10:d={duration_s}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-tune", "stillimage",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


@pytest.fixture
def instructional_dir(tmp_path: Path) -> Path:
    d = tmp_path / "Foo by Bar"
    d.mkdir()
    _make_video(d / "Foo by Bar1.mp4", duration_s=30)
    _make_video(d / "Foo by Bar2.mp4", duration_s=20)
    return d


def _make_oracle(durations: dict[int, list[tuple[float, float, str]]]) -> OracleResult:
    volumes = []
    for num, chapters in durations.items():
        ch_objs = [
            OracleChapter(title=t, start_s=s, end_s=e) for (s, e, t) in chapters
        ]
        total = ch_objs[-1].end_s
        volumes.append(
            OracleVolume(number=num, chapters=ch_objs, total_duration_s=total)
        )
    return OracleResult(
        product_url="https://example.com/p",
        provider_id="test",
        volumes=volumes,
    )


def test_split_creates_seasons_and_files(instructional_dir: Path) -> None:
    oracle = _make_oracle({
        1: [(0.0, 10.0, "Intro"), (10.0, 20.0, "Mid"), (20.0, 30.0, "End")],
        2: [(0.0, 10.0, "Alpha"), (10.0, 20.0, "Beta")],
    })
    calls: list[tuple[float, str]] = []

    splitter = OracleSplitter(instructional_dir, oracle)
    report = splitter.split(progress_cb=lambda p, m: calls.append((p, m)))

    assert isinstance(report, SplitReport)
    assert report.volumes_processed == 2
    assert report.chapters_created == 5

    s1 = list((instructional_dir / "Season 01").glob("*.mp4"))
    s2 = list((instructional_dir / "Season 02").glob("*.mp4"))
    assert len(s1) == 3
    assert len(s2) == 2

    assert any(f.name.startswith("S01E01") for f in s1)
    assert any(f.name.startswith("S02E02") for f in s2)

    # progress_cb invoked at least total_chapters times, ends at 100
    assert len(calls) >= 5
    assert calls[-1][0] == pytest.approx(100.0)


def test_missing_volume_mp4_warns_and_skips(instructional_dir: Path) -> None:
    oracle = _make_oracle({
        1: [(0.0, 10.0, "A")],
        2: [(0.0, 10.0, "B")],
        3: [(0.0, 5.0, "Ghost")],  # no mp4 ending in 3
    })

    splitter = OracleSplitter(instructional_dir, oracle)
    report = splitter.split()

    assert report.volumes_processed == 2  # only 1 and 2
    assert report.chapters_created == 2
    assert any("Volume 3" in w for w in report.warnings)
    assert not (instructional_dir / "Season 03").exists()


def test_duration_mismatch_flags_needs_review(
    instructional_dir: Path,
) -> None:
    # mp4 of vol 1 is 30s; oracle claims 60s -> diff > 5s -> needs_review
    oracle = _make_oracle({
        1: [(0.0, 30.0, "A"), (30.0, 60.0, "B")],
        2: [(0.0, 20.0, "C")],
    })
    splitter = OracleSplitter(instructional_dir, oracle)
    report = splitter.split()

    assert 1 in report.needs_review_flags
    assert 2 not in report.needs_review_flags
    assert any("Volume 1" in w and "differs" in w for w in report.warnings)
