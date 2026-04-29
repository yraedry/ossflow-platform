"""Lightweight per-instructional refresh of filesystem-derived flags.

The scan cache stores per-video booleans (has_subtitles_en, has_dubbing, ...)
captured during the last ``scan_library`` pass. Over time, these drift: user
deletes a ``.srt`` outside the UI, dubbing worker writes a new ``*_DOBLADO.mkv``,
poster gets added manually, etc.

Rather than re-walking the whole library (expensive on NAS/CIFS), this module
re-stats only the sidecar files for videos that are already known, mutating
the cached entries in place. Cost: a handful of ``stat()`` calls per video,
no ``iterdir``.

Two entry points:

- :func:`refresh_instructional_flags` — refresh one cached instructional.
- :func:`rediscover_instructional` — re-``iterdir`` a single folder to pick up
  newly added / removed videos in an existing instructional.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

import subprocess

from api.scan_cache import find_poster

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _probe_duration(path: Path) -> float | None:
    """Fast ffprobe to get only duration in seconds."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:
        pass
    return None


# Spanish language tags emitted by various muxers (ffmpeg, MakeMKV, mp4box…).
# We normalise to lowercase before checking. ``""`` (untagged) explicitly does
# NOT count — many original instructionals ship audio with no language tag and
# we'd false-positive every chapter as already-dubbed.
_SPANISH_LANG_TAGS = frozenset({"spa", "es", "esp", "spanish", "español"})
_ENGLISH_LANG_TAGS = frozenset({"eng", "en", "english"})


def _probe_track_languages(path: Path) -> tuple[list[str], list[str]]:
    """Return (audio_languages, subtitle_languages) for *path*, both lowercased.

    Single ffprobe invocation (JSON output) covering audio + subtitle streams
    so we don't pay double on libraries with thousands of chapters. Untagged
    streams come back as ``""``; "und" (undefined) is also normalised to "".

    Cost: ~30-80 ms per call cold. Cached per-video by (mtime, size)
    fingerprint in library.json — see ``_video_flags``.
    """
    import json
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type:stream_tags=language",
                "-of", "json",
                str(path),
            ],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return [], []
        data = json.loads(r.stdout or "{}")
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, OSError):
        return [], []
    audio: list[str] = []
    subs: list[str] = []
    for s in data.get("streams", []):
        codec_type = (s.get("codec_type") or "").lower()
        tag = ((s.get("tags") or {}).get("language") or "").strip().lower()
        if tag == "und":
            tag = ""
        if codec_type == "audio":
            audio.append(tag)
        elif codec_type == "subtitle":
            subs.append(tag)
    return audio, subs


def _has_spanish_audio(audio_languages: list[str] | None) -> bool:
    """True if any cached audio language is a recognised Spanish tag."""
    if not audio_languages:
        return False
    return any(lang in _SPANISH_LANG_TAGS for lang in audio_languages)


def _has_english_subtitle(subtitle_languages: list[str] | None) -> bool:
    """True if any cached subtitle language is a recognised English tag."""
    if not subtitle_languages:
        return False
    return any(lang in _ENGLISH_LANG_TAGS for lang in subtitle_languages)


def _has_spanish_subtitle(subtitle_languages: list[str] | None) -> bool:
    if not subtitle_languages:
        return False
    return any(lang in _SPANISH_LANG_TAGS for lang in subtitle_languages)


def _file_fingerprint(path: Path) -> tuple[int, int] | None:
    """Cheap (mtime_int, size) tuple used to invalidate the audio_languages
    cache when the file is rewritten (e.g. after a promotion remux)."""
    try:
        st = path.stat()
        return (int(st.st_mtime), st.st_size)
    except OSError:
        return None


_DUB_SUFFIXES = ("_DOBLADO.mkv", "_DOBLADO.mp4")
_CHAPTER_RE = re.compile(r"S\d{2}E\d{2}")


def _video_flags(video_path: Path, cached: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-stat sidecars for one video. Returns fields to merge onto the cached entry.

    ``cached`` is the previous entry from library.json (if any). We reuse the
    ffprobe result (audio + subtitle stream languages) when the file
    fingerprint is unchanged — ffprobe is cheap individually but the library
    has thousands of chapters and probing every one on every refresh would
    dominate cost.

    Embedded streams matter because the "promote" workflow ends with a
    multi-track .mkv whose external .srt/.mp4 inputs were deleted. Without
    embedded-track detection the UI checks would all flip back to false.
    """
    folder = video_path.parent
    base = video_path.stem
    has_en_srt_sidecar = (
        (folder / f"{base}.en.srt").exists() or (folder / f"{base}.srt").exists()
    )
    # Literal ES subtitles. The dubbing-adapted file is {base}.dub.es.srt
    # (different stem), so this check unambiguously matches only the literal.
    has_es_srt_sidecar = (
        (folder / f"{base}.es.srt").exists()
        or (folder / f"{base}.ES.srt").exists()
        or (folder / f"{base}_ES.srt").exists()
        or (folder / f"{base}_ESP_DUB.srt").exists()
    )

    # ffprobe — covers BOTH audio (for has_dubbing post-promote) and
    # subtitle (for has_subtitles_en/es post-promote) streams in a single
    # invocation. Cached by (mtime, size) fingerprint.
    fingerprint = _file_fingerprint(video_path)
    cached_fp = None
    audio_languages: list[str] | None = None
    subtitle_languages: list[str] | None = None
    if cached is not None:
        cached_fp_raw = cached.get("audio_fingerprint")
        if isinstance(cached_fp_raw, list) and len(cached_fp_raw) == 2:
            cached_fp = (int(cached_fp_raw[0]), int(cached_fp_raw[1]))
        cached_audio = cached.get("audio_languages")
        if isinstance(cached_audio, list):
            audio_languages = [str(x) for x in cached_audio]
        cached_subs = cached.get("subtitle_languages")
        if isinstance(cached_subs, list):
            subtitle_languages = [str(x) for x in cached_subs]
    needs_probe = (
        fingerprint is not None
        and (fingerprint != cached_fp or subtitle_languages is None)
    )
    if needs_probe:
        audio_languages, subtitle_languages = _probe_track_languages(video_path)

    # Dub present if any of:
    #   - legacy XTTS:  <name>_DOBLADO.mkv/mp4 next to source
    #   - flujo B v5:   <folder>/doblajes/<name>.mkv  (pre-promotion)
    #   - Studio E2E:   <folder>/elevenlabs/<source.ext>
    #   - promoted:     audio_languages contains a Spanish tag
    has_dubbed = (
        any((folder / f"{base}{sfx}").exists() for sfx in _DUB_SUFFIXES)
        or (folder / "doblajes" / f"{base}.mkv").exists()
        or (folder / "elevenlabs" / video_path.name).exists()
        or _has_spanish_audio(audio_languages)
    )

    # Subtitle detection: external sidecar OR embedded stream with the
    # right language tag. Only the embedded path survives promotion.
    has_en_subs = has_en_srt_sidecar or _has_english_subtitle(subtitle_languages)
    has_es_subs = has_es_srt_sidecar or _has_spanish_subtitle(subtitle_languages)

    # is_promoted: file is the multi-track final form (Spanish embedded AND
    # no sidecar in doblajes/). Used by the frontend to hide the "Promover"
    # button on rows that are already done.
    is_promoted = (
        _has_spanish_audio(audio_languages)
        and not (folder / "doblajes" / f"{base}.mkv").exists()
    )

    try:
        size_mb = round(video_path.stat().st_size / (1024 * 1024), 1)
    except OSError:
        size_mb = None

    out: dict[str, Any] = {
        "has_subtitles_en": has_en_subs,
        "has_subtitles_es": has_es_subs,
        "has_dubbing": has_dubbed,
        "is_promoted": is_promoted,
        "is_chapter": bool(_CHAPTER_RE.search(video_path.name)),
    }
    if audio_languages is not None:
        out["audio_languages"] = audio_languages
    if subtitle_languages is not None:
        out["subtitle_languages"] = subtitle_languages
    if fingerprint is not None:
        out["audio_fingerprint"] = list(fingerprint)
    if size_mb is not None:
        out["size_mb"] = size_mb
    return out


def ensure_duration(video: dict[str, Any]) -> None:
    """Populate ``duration`` via ffprobe if missing from a cached video entry."""
    if video.get("duration") is not None:
        return
    vp = video.get("path")
    if vp:
        video["duration"] = _probe_duration(Path(vp))


def refresh_instructional_flags(item: dict[str, Any]) -> dict[str, Any]:
    """Mutate cached instructional in place: re-stat poster + per-video flags.

    Drops videos whose file no longer exists on disk. Returns the same dict.
    """
    folder_str = item.get("path")
    if not folder_str:
        return item
    folder = Path(folder_str)

    poster = find_poster(folder)
    item["has_poster"] = poster is not None
    item["poster_filename"] = poster.name if poster else None
    if poster is not None:
        try:
            item["poster_mtime"] = int(poster.stat().st_mtime)
        except OSError:
            item["poster_mtime"] = None
    else:
        item["poster_mtime"] = None

    videos = item.get("videos") or []
    fresh_videos: list[dict[str, Any]] = []
    subtitled = dubbed = chapters = 0
    for v in videos:
        if not isinstance(v, dict):
            continue
        vp_str = v.get("path")
        if not vp_str:
            continue
        vp = Path(vp_str)
        if not vp.exists():
            continue
        # Drop stale `*_DOBLADO.mkv/mp4` entries left in cache from previous
        # scans (before the filter was added to scan_library).
        if vp.stem.endswith("_DOBLADO"):
            continue
        v.update(_video_flags(vp, cached=v))
        fresh_videos.append(v)
        if v.get("has_subtitles_en"):
            subtitled += 1
        if v.get("has_dubbing"):
            dubbed += 1
        if v.get("is_chapter"):
            chapters += 1

    item["videos"] = fresh_videos
    item["total_videos"] = len(fresh_videos)
    item["subtitled"] = subtitled
    item["dubbed"] = dubbed
    item["chapters_detected"] = chapters
    return item


def rediscover_instructional(item: dict[str, Any]) -> dict[str, Any]:
    """Re-walk the instructional folder to pick up added/removed video files.

    Preserves per-video fields already in cache for files that still exist
    (duration, etc.), adds entries for new videos, drops entries for deleted ones.
    Then delegates to :func:`refresh_instructional_flags` for sidecar re-stat.
    """
    folder_str = item.get("path")
    if not folder_str:
        return item
    folder = Path(folder_str)
    if not folder.exists() or not folder.is_dir():
        return item

    existing_by_path = {v.get("path"): v for v in (item.get("videos") or []) if isinstance(v, dict)}
    discovered: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(folder):
        # Skip artefact folders (see scan_library for rationale). In-place
        # mutation on dirnames prunes the walk so ``elevenlabs/`` and
        # ``doblajes/`` contents never surface as duplicated videos.
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in ("elevenlabs", "doblajes")
        ]
        dp = Path(dirpath)
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            # Skip dubbed outputs — covers "name_DOBLADO.ext" and any
            # backup variant ("name_DOBLADO_B_v5.mkv").
            if "_DOBLADO" in Path(fn).stem:
                continue
            vp = dp / fn
            key = str(vp)
            cached = existing_by_path.get(key)
            if cached is not None:
                discovered.append(cached)
            else:
                discovered.append({
                    "filename": fn,
                    "path": key,
                    "duration": None,  # lazy — filled by ensure_duration later
                })

    item["videos"] = discovered
    return refresh_instructional_flags(item)


def refresh_many(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply :func:`refresh_instructional_flags` to a sequence of cached items."""
    out: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(refresh_instructional_flags(it))
    return out
