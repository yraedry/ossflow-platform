"""Chapter (per-video) operations: currently supports renaming a chapter file
while preserving the SNNeMM prefix and keeping sibling files (subs, dubs) in sync.

Single responsibility: expose one HTTP surface to rename the main video plus
all its sidecar files (same stem) atomically enough for our use case.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.settings import get_library_path

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chapters", tags=["chapters"])

# Regex to split `{prefix} - SNNeMM - {title}{ext}` on the filename (not path).
# Example: "John Danaher - S01E03 - Armbar Fundamentals.mkv"
#          prefix = "John Danaher", season=01, ep=03, ext=".mkv"
_SNNEMM_RE = re.compile(
    r"^(?P<prefix>.*?)\s*-\s*S(?P<season>\d{2})E(?P<ep>\d{2,3})\s*-\s*.*(?P<ext>\.[^.]+)$"
)

# Characters illegal on Windows filenames (and that we never want anywhere).
_ILLEGAL_RE = re.compile(r'[\/\\:*?"<>|]')
_WS_RE = re.compile(r"\s+")

# Sidecar suffixes we may need to rename alongside the main video.
# Each entry is the suffix that replaces the video extension entirely
# (so "Name.mkv" → "Name.srt" / "Name.en.srt" / ...).
_SIDECAR_SUFFIXES = (
    ".srt",
    ".en.srt",
    ".es.srt",
    ".ES.srt",
    "_ESP_DUB.srt",
    "_DOBLADO.mkv",
    "_DOBLADO.mp4",
)


def _sanitize_title(raw: str) -> str:
    """Strip, replace illegal chars with `_`, collapse whitespace, cap at 120.

    Returns empty string if the result is empty / only whitespace; callers
    must treat that as a validation failure.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    s = _ILLEGAL_RE.sub("_", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    if len(s) > 120:
        s = s[:120].rstrip()
    return s


def _resolve_within_library(candidate: Path, library_root: Path) -> Path:
    """Return the resolved absolute path, or raise 403 if it escapes root."""
    try:
        resolved = candidate.resolve(strict=False)
        root_resolved = library_root.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise HTTPException(status_code=403, detail=f"Path traversal: {e}")

    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Path traversal: target escapes library_path",
        )
    return resolved


def _find_sibling(stem_path: Path, suffix: str) -> Path | None:
    """Return existing sibling path with ``suffix`` replacing the full ext, else None."""
    candidate = stem_path.with_name(stem_path.stem + suffix)
    return candidate if candidate.exists() else None


@router.post("/rename-by-oracle")
async def rename_season_by_oracle(request: Request) -> Any:
    """Rename all chapters in a Season folder using oracle chapter titles.

    Matches each file's SNNeMM episode code to the oracle's (volume, episode)
    and replaces the title portion.  Files with no oracle match are skipped.

    Body: {"season_path": "...", "oracle": {<OracleResult>}}.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid JSON body")

    season_path_str = body.get("season_path")
    oracle = body.get("oracle")

    if not isinstance(season_path_str, str) or not season_path_str:
        raise HTTPException(status_code=422, detail="season_path is required")
    if not isinstance(oracle, dict):
        raise HTTPException(status_code=422, detail="oracle is required")

    library_root_str = get_library_path()
    if not library_root_str:
        raise HTTPException(status_code=400, detail="library_path not configured")
    library_root = Path(library_root_str)

    from api.paths import to_container_path
    container_path_str = to_container_path(season_path_str, library_root_str)
    season_dir = Path(container_path_str)
    resolved_season = _resolve_within_library(season_dir, library_root)
    if not resolved_season.exists() or not resolved_season.is_dir():
        raise HTTPException(status_code=404, detail=f"Season directory not found: {container_path_str}")

    # Build a mapping (volume_num, episode_num) → title from oracle volumes.
    # oracle structure: {"volumes": [{"number": N, "chapters": [{"number": M, "title": "..."}]}]}
    oracle_map: dict[tuple[int, int], str] = {}
    for vol in oracle.get("volumes", []):
        vol_num = int(vol.get("number", 0))
        for idx, ch in enumerate(vol.get("chapters", []), start=1):
            ep_num = int(ch.get("number", idx))
            title = ch.get("title", "").strip()
            if title:
                oracle_map[(vol_num, ep_num)] = title

    if not oracle_map:
        raise HTTPException(status_code=422, detail="Oracle contains no chapter titles")

    VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov"}
    renamed: list[dict[str, str]] = []
    skipped: list[str] = []

    # Formato alternativo: "1-2.mp4" → vol=1, ep=2 (archivos sin renombrar del splitter)
    _RAW_RE = re.compile(r"^(?P<vol>\d+)-(?P<ep>\d+)(?P<ext>\.[^.]+)$")
    instructional_name = body.get("instructional_name", "").strip()

    for f in sorted(resolved_season.iterdir()):
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXTS:
            continue

        m = _SNNEMM_RE.match(f.name)
        if m:
            season_num = int(m.group("season"))
            ep_num = int(m.group("ep"))
            prefix = m.group("prefix").strip()
            season_str = m.group("season")
            ep_str = m.group("ep")
            ext = m.group("ext")
        else:
            m2 = _RAW_RE.match(f.name)
            if not m2:
                skipped.append(f.name)
                continue
            season_num = int(m2.group("vol"))
            ep_num = int(m2.group("ep"))
            prefix = instructional_name
            season_str = f"{season_num:02d}"
            ep_str = f"{ep_num:02d}"
            ext = m2.group("ext")

        oracle_title = oracle_map.get((season_num, ep_num))
        if oracle_title is None:
            skipped.append(f.name)
            continue

        sanitized = _sanitize_title(oracle_title)
        if not sanitized:
            skipped.append(f.name)
            continue

        sep = f" - " if prefix else ""
        new_filename = f"{prefix}{sep}S{season_str}E{ep_str} - {sanitized}{ext}"
        new_path = f.with_name(new_filename)

        if new_path == f:
            continue
        if new_path.exists():
            log.warning("rename-by-oracle: target exists, skipping: %s", new_path.name)
            skipped.append(f.name)
            continue

        _resolve_within_library(new_path, library_root)
        old_stem = f.stem
        new_stem = new_path.stem

        os.rename(f, new_path)
        renamed.append({"from": str(f), "to": str(new_path)})

        for suffix in _SIDECAR_SUFFIXES:
            sib_old = f.with_name(old_stem + suffix)
            if not sib_old.exists():
                continue
            sib_new = f.with_name(new_stem + suffix)
            if sib_new == sib_old or sib_new.exists():
                continue
            os.rename(sib_old, sib_new)
            renamed.append({"from": str(sib_old), "to": str(sib_new)})

    return JSONResponse({"renamed": renamed, "skipped": skipped})


@router.patch("/rename")
async def rename_chapter(request: Request) -> Any:
    """Rename a chapter file (and its sidecars) preserving the SNNeMM prefix.

    Body: {"old_path": "...", "new_title": "..."}.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Body must be an object")

    old_path = body.get("old_path")
    new_title_raw = body.get("new_title")

    if not isinstance(old_path, str) or not old_path:
        raise HTTPException(status_code=422, detail="old_path is required")
    if not isinstance(new_title_raw, str):
        raise HTTPException(status_code=422, detail="new_title is required")

    sanitized = _sanitize_title(new_title_raw)
    if not sanitized:
        raise HTTPException(
            status_code=422,
            detail="new_title is empty after sanitization",
        )

    library_root_str = get_library_path()
    if not library_root_str:
        raise HTTPException(status_code=400, detail="library_path not configured")
    library_root = Path(library_root_str)

    old = Path(old_path)
    resolved_old = _resolve_within_library(old, library_root)

    if not resolved_old.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {old_path}")
    if not resolved_old.is_file():
        raise HTTPException(status_code=404, detail="old_path is not a file")

    # Parse SNNeMM from the filename (not from full path).
    m = _SNNEMM_RE.match(resolved_old.name)
    if not m:
        raise HTTPException(
            status_code=422,
            detail="Filename does not match `{prefix} - SNNeMM - {title}{ext}` pattern",
        )

    prefix = m.group("prefix").strip()
    season = m.group("season")
    ep = m.group("ep")
    ext = m.group("ext")

    new_filename = f"{prefix} - S{season}E{ep} - {sanitized}{ext}"
    new_path = resolved_old.with_name(new_filename)
    # Ensure resulting path still lives inside the library (redundant but cheap).
    _resolve_within_library(new_path, library_root)

    renamed: list[dict[str, str]] = []

    # Rename main file first (idempotent if name unchanged).
    if new_path != resolved_old:
        if new_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Target already exists: {new_path.name}",
            )
        os.rename(resolved_old, new_path)
    renamed.append({"from": str(resolved_old), "to": str(new_path)})

    # Rename sidecars (based on original stem → new stem).
    old_stem = resolved_old.stem  # e.g. "Author - S01E01 - Old Title"
    new_stem = new_path.stem
    for suffix in _SIDECAR_SUFFIXES:
        # Siblings were found via the OLD stem in the OLD location.
        sib_old = resolved_old.with_name(old_stem + suffix)
        if not sib_old.exists():
            continue
        sib_new = resolved_old.with_name(new_stem + suffix)
        if sib_new == sib_old:
            continue
        if sib_new.exists():
            log.warning("Sidecar target already exists, skipping: %s", sib_new)
            continue
        os.rename(sib_old, sib_new)
        renamed.append({"from": str(sib_old), "to": str(sib_new)})

    return JSONResponse({"renamed": renamed})


# WIRE_ROUTER: from api.chapters import router as chapters_router; app.include_router(chapters_router)
