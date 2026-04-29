"""Promote a dubbed video to its final multi-track form.

After the dubbing pipeline finishes, ``<Season>/doblajes/<name>.mkv`` holds
the dubbed-only video. The user reviews it, then "promotes" it: a single
``<Season>/<name>.mkv`` is built containing the dubbed Spanish audio (default
track) + the original English audio + Spanish/English subtitles, and the
intermediate artefacts are deleted.

Atomic strategy: write to ``<name>.mkv.tmp`` first; only ``os.replace`` into
the final name AFTER ffmpeg succeeds; only delete inputs AFTER the rename.
A crash mid-mux leaves the chapter intact (originals still on disk, no
``.mkv`` collision).

Detection survives the operation: ``library_refresh._video_flags`` calls
ffprobe on the merged ``.mkv``, sees a ``spa`` audio stream, and keeps
``has_dubbing=True`` on subsequent scans.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/promote", tags=["promote"])


_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")

# Per-season locks so two concurrent promotions on different seasons don't
# block each other but a single season is never processed in parallel
# (sequential remuxes — NAS I/O hates concurrent ffmpeg).
_season_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _season_lock(season_dir: Path) -> threading.Lock:
    key = str(season_dir)
    with _locks_guard:
        lock = _season_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _season_locks[key] = lock
        return lock


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PromoteChapterBody(BaseModel):
    video_path: str  # host path of the ORIGINAL (e.g. <Season>/<name>.mp4)


class PromoteSeasonBody(BaseModel):
    season_path: str


@dataclass
class _Inputs:
    """Resolved file paths for one chapter promotion."""
    original: Path           # <Season>/<name>.mp4 (or .mkv, etc.)
    dubbed: Path             # <Season>/doblajes/<name>.mkv
    output: Path             # <Season>/<name>.mkv (final)
    output_tmp: Path         # <Season>/<name>.mkv.tmp
    es_srt: Optional[Path]
    en_srt: Optional[Path]
    sidecars_to_delete: list[Path]


def _resolve_inputs(original_path: str) -> _Inputs:
    original = Path(original_path)
    if not original.is_file():
        raise HTTPException(status_code=409, detail={
            "code": "original_missing",
            "message": f"Original video not found: {original}",
        })
    season = original.parent
    base = original.stem
    dubbed = season / "doblajes" / f"{base}.mkv"
    if not dubbed.is_file():
        raise HTTPException(status_code=409, detail={
            "code": "dubbed_missing",
            "message": f"Dubbed file not found: {dubbed}",
        })
    output = season / f"{base}.mkv"
    if output.exists() and output.resolve() != original.resolve():
        # A separate <name>.mkv already lives next to the original .mp4 —
        # the user (or a previous run) may have left a partial result here.
        # Refuse rather than overwrite; manual cleanup required.
        raise HTTPException(status_code=409, detail={
            "code": "already_promoted",
            "message": (
                f"Output collision: {output} already exists. Move or delete it "
                "before promoting."
            ),
        })
    output_tmp = season / f"{base}.mkv.tmp"

    def _existing(*candidates: Path) -> Optional[Path]:
        for c in candidates:
            if c.exists():
                return c
        return None

    es_srt = _existing(
        season / f"{base}.es.srt",
        season / f"{base}.ES.srt",
        season / f"{base}_ES.srt",
    )
    en_srt = _existing(
        season / f"{base}.en.srt",
        season / f"{base}.srt",
    )

    # Files we'll remove after a successful mux. .bjj-meta.json is at the
    # instructional root (not per-chapter) — never touched.
    sidecars = [
        season / f"{base}.dub.es.srt",
        season / f"{base}_VOCALS.wav",
        season / f"{base}_BACKGROUND.wav",
        season / f"{base}_ref.wav",
        season / f"{base}_AUDIO_ESP.wav",
        season / f"{base}_QA_TMP.wav",
        season / f"{base}.words.json",
        season / f"{base}.dub-qa.json",
    ]
    if es_srt is not None:
        sidecars.append(es_srt)
    if en_srt is not None:
        sidecars.append(en_srt)

    return _Inputs(
        original=original, dubbed=dubbed,
        output=output, output_tmp=output_tmp,
        es_srt=es_srt, en_srt=en_srt,
        sidecars_to_delete=sidecars,
    )


# ---------------------------------------------------------------------------
# Pipeline guard
# ---------------------------------------------------------------------------


def _pipeline_active_for(paths: list[Path]) -> Optional[str]:
    """Return the pipeline_id of any RUNNING pipeline whose target intersects
    *paths*, or None. Lazy import avoids circular at module load."""
    try:
        from api.pipeline import _pipelines, StepStatus
    except ImportError:
        return None
    targets = {str(p.resolve()) for p in paths}
    for pid, pipe in _pipelines.items():
        if pipe.status != StepStatus.RUNNING:
            continue
        try:
            ppath = str(Path(pipe.path).resolve())
        except OSError:
            continue
        if ppath in targets:
            return pid
        # Pipeline pointed at the season folder also blocks per-chapter ops.
        for t in targets:
            if t.startswith(ppath + "/") or t.startswith(ppath + "\\"):
                return pid
    return None


# ---------------------------------------------------------------------------
# ffmpeg invocation
# ---------------------------------------------------------------------------


def _build_ffmpeg_argv(inp: _Inputs) -> list[str]:
    """Construct the ffmpeg command for the multi-track mux.

    Stream layout in output:
      v:0  ← dubbed video      (input 0, stream copy)
      a:0  ← Spanish dubbed    (input 0, default disposition)
      a:1  ← English original  (input 1)
      s:0  ← Spanish .srt      (input 2 if es_srt) — default
      s:1  ← English .srt      (input 3 if en_srt)
    """
    argv: list[str] = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(inp.dubbed),
        "-i", str(inp.original),
    ]
    sub_inputs: list[Path] = []
    if inp.es_srt is not None:
        argv += ["-i", str(inp.es_srt)]
        sub_inputs.append(inp.es_srt)
    if inp.en_srt is not None:
        argv += ["-i", str(inp.en_srt)]
        sub_inputs.append(inp.en_srt)

    # Maps. `?` makes the map optional (no error if the audio stream is
    # absent in input 1 — extremely unlikely but cheap insurance).
    argv += ["-map", "0:v:0", "-map", "0:a:0?", "-map", "1:a:0?"]
    for idx, _ in enumerate(sub_inputs, start=2):
        argv += ["-map", f"{idx}:0?"]

    # Stream copy — no transcoding. Both audios are AAC from the dubbing
    # pipeline / source; subs are SRT and MKV holds them natively.
    argv += ["-c:v", "copy", "-c:a", "copy", "-c:s", "copy"]

    # Audio language + title metadata.
    argv += [
        "-metadata:s:a:0", "language=spa",
        "-metadata:s:a:0", "title=Español (doblaje IA)",
        "-metadata:s:a:1", "language=eng",
        "-metadata:s:a:1", "title=English (original)",
    ]
    argv += ["-disposition:a:0", "default", "-disposition:a:1", "0"]

    # Subtitle metadata follows the order of sub_inputs.
    for s_idx, src in enumerate(sub_inputs):
        if src is inp.es_srt:
            argv += [
                f"-metadata:s:s:{s_idx}", "language=spa",
                f"-metadata:s:s:{s_idx}", "title=Español",
                f"-disposition:s:{s_idx}", "default",
            ]
        else:  # en_srt
            argv += [
                f"-metadata:s:s:{s_idx}", "language=eng",
                f"-metadata:s:s:{s_idx}", "title=English",
                f"-disposition:s:{s_idx}", "0",
            ]

    # Force matroska format — ffmpeg can't infer it from the .tmp extension.
    argv += ["-f", "matroska", str(inp.output_tmp)]
    return argv


def _run_ffmpeg(argv: list[str]) -> None:
    """Invoke ffmpeg, raise HTTPException(500) with stderr tail on failure."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail={
            "code": "ffmpeg_missing",
            "message": f"ffmpeg not in PATH: {exc}",
        })
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail={
            "code": "ffmpeg_timeout",
            "message": "ffmpeg exceeded 600 s — file may be huge or stuck",
        })
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-15:]
        log.error("ffmpeg failed (rc=%d):\n%s", r.returncode, "\n".join(tail))
        raise HTTPException(status_code=500, detail={
            "code": "ffmpeg_failed",
            "message": "ffmpeg returned non-zero",
            "stderr_tail": tail,
        })


# ---------------------------------------------------------------------------
# One-chapter promotion (used by both endpoints)
# ---------------------------------------------------------------------------


def _promote_one(original_path: str) -> dict:
    """Mux + cleanup for a single chapter. Raises HTTPException on errors."""
    inp = _resolve_inputs(original_path)

    active = _pipeline_active_for([inp.original, inp.dubbed])
    if active:
        raise HTTPException(status_code=409, detail={
            "code": "pipeline_active",
            "message": f"Pipeline {active} is running on this video",
        })

    season = inp.original.parent
    with _season_lock(season):
        argv = _build_ffmpeg_argv(inp)
        log.info("Promoting %s → %s  cmd: %s", inp.dubbed.name, inp.output.name, " ".join(argv))
        try:
            _run_ffmpeg(argv)
            # Atomic swap: replace any leftover temp first (we already verified
            # `inp.output` does not exist or is the original itself).
            inp.output_tmp.replace(inp.output)
        except Exception:
            # ffmpeg failed or replace exploded — sweep the .tmp away so the
            # next call doesn't trip on stale state.
            try:
                if inp.output_tmp.exists():
                    inp.output_tmp.unlink()
            except OSError:
                pass
            raise

        # Mux succeeded. Now delete the inputs. Order matters:
        #   1. Original video (the .mp4 that may share stem with output .mkv).
        #      If output and original have the same stem but different
        #      extensions we MUST unlink the original here; if same exact
        #      path (.mkv original), the .replace above already replaced it.
        deleted: list[str] = []
        if inp.original.exists() and inp.original.resolve() != inp.output.resolve():
            try:
                inp.original.unlink()
                deleted.append(str(inp.original))
            except OSError as exc:
                log.warning("Could not delete original %s: %s", inp.original, exc)

        # 2. Dubbed file in doblajes/, then the doblajes/ folder if empty.
        try:
            inp.dubbed.unlink()
            deleted.append(str(inp.dubbed))
        except OSError as exc:
            log.warning("Could not delete dubbed %s: %s", inp.dubbed, exc)
        try:
            doblajes_dir = inp.dubbed.parent
            if doblajes_dir.is_dir() and not any(doblajes_dir.iterdir()):
                doblajes_dir.rmdir()
                deleted.append(str(doblajes_dir))
        except OSError:
            pass

        # 3. Sidecars (best effort; any failure is logged but not surfaced).
        for sc in inp.sidecars_to_delete:
            try:
                if sc.exists():
                    sc.unlink()
                    deleted.append(str(sc))
            except OSError as exc:
                log.warning("Could not delete sidecar %s: %s", sc, exc)

    # Refresh the cached entry for this instructional so the UI reflects
    # the new state without waiting for a manual rescan.
    _refresh_cache_for(inp.output)

    return {
        "ok": True,
        "output_path": str(inp.output),
        "deleted": deleted,
        "muxed_streams": {
            "audio": ["spa", "eng"],
            "subs": [
                tag for tag, src in (("spa", inp.es_srt), ("eng", inp.en_srt))
                if src is not None
            ],
        },
    }


def _refresh_cache_for(video_path: Path) -> None:
    """Re-stat the instructional containing *video_path* so the new
    .mkv (with embedded ES audio) replaces the cached entry promptly."""
    try:
        from api.scan_cache import ScanCache
        from api.settings import CONFIG_DIR, get_library_path
        from api.library_refresh import refresh_instructional_flags

        cache = ScanCache(CONFIG_DIR / "library.json")
        data = cache.load()
        if not data:
            return
        items = data.get("instructionals", []) if isinstance(data, dict) else []
        # Walk up from the video to find the instructional root (direct child
        # of library_path).
        lib = get_library_path()
        folder = video_path.parent
        if lib:
            lib_p = Path(lib)
            while folder.parent != lib_p and folder.parent != folder:
                folder = folder.parent
        folder_str = str(folder)
        match = next(
            (it for it in items if it.get("path") == folder_str),
            None,
        )
        if match is None:
            return
        refresh_instructional_flags(match)
        cache.save(items)
    except Exception:  # noqa: BLE001
        log.warning("Could not refresh cache after promote", exc_info=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/chapter")
def promote_chapter(body: PromoteChapterBody) -> dict:
    """Promote a single dubbed chapter to its final multi-track form."""
    return _promote_one(body.video_path)


@router.post("/season")
def promote_season(body: PromoteSeasonBody) -> dict:
    """Promote every dubbed chapter under *season_path*. Sequential."""
    season = Path(body.season_path)
    if not season.is_dir():
        raise HTTPException(status_code=409, detail={
            "code": "season_missing",
            "message": f"Season folder not found: {season}",
        })
    doblajes = season / "doblajes"
    if not doblajes.is_dir():
        return {"promoted": [], "skipped": [], "failed": [],
                "message": "Nothing to promote — no doblajes/ folder"}

    candidates: list[Path] = []
    for f in sorted(season.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _VIDEO_EXTS:
            continue
        if (doblajes / f"{f.stem}.mkv").exists():
            candidates.append(f)

    promoted: list[dict] = []
    skipped: list[dict] = []
    failed: list[dict] = []
    for orig in candidates:
        try:
            result = _promote_one(str(orig))
            promoted.append({"path": str(orig), "output": result["output_path"]})
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
            bucket = skipped if exc.status_code == 409 else failed
            bucket.append({"path": str(orig), **detail})
        except Exception as exc:  # noqa: BLE001
            log.exception("Promote failed for %s", orig)
            failed.append({"path": str(orig), "code": "unexpected", "message": str(exc)})

    return {
        "promoted": [p["path"] for p in promoted],
        "skipped": skipped,
        "failed": failed,
        "promoted_count": len(promoted),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
    }


# Re-export shutil so unit tests can monkeypatch the module without
# rewriting the import chain. Keeps a single place to stub.
__all__ = ["router", "_promote_one", "_build_ffmpeg_argv", "_resolve_inputs", "shutil"]
