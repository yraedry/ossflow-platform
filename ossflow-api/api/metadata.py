"""Sidecar JSON metadata editor for library instructionals.

Stores `.bjj-meta.json` at the root of each instructional folder with
user-editable fields: instructor, topic, tags, synopsis, year.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.settings import get_library_path

router = APIRouter(prefix="/api/library", tags=["metadata"])

SIDECAR_NAME = ".bjj-meta.json"

DEFAULT_METADATA: dict[str, Any] = {
    "instructor": "",
    "topic": "",
    "tags": [],
    "synopsis": "",
    "year": None,
    # voice_profile: filename under /voices (e.g. "narrador_es.wav") or "" to
    # clone the instructor's own voice. Read by pipeline.py when queuing the
    # dubbing step for this instructional.
    "voice_profile": "",
}


def _resolve_target(name: str) -> Path:
    """Resolve instructional folder with anti path-traversal checks.

    Raises HTTPException on invalid input; returns the Path otherwise.
    """
    lib = get_library_path()
    if not lib:
        raise HTTPException(status_code=404, detail="library_path not configured")

    base = Path(lib).resolve()
    try:
        target = (base / name).resolve()
    except OSError:
        raise HTTPException(status_code=403, detail="invalid path")

    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=403, detail="path traversal denied")
    if target == base:
        raise HTTPException(status_code=403, detail="invalid target")
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail="instructional not found")
    return target


def _validate_payload(data: Any) -> dict[str, Any]:
    """Validate body types. Raises HTTPException on any type mismatch."""
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="body must be a JSON object")

    instructor = data.get("instructor", "")
    topic = data.get("topic", "")
    tags = data.get("tags", [])
    synopsis = data.get("synopsis", "")
    year = data.get("year", None)
    voice_profile = data.get("voice_profile", "")

    if not isinstance(instructor, str):
        raise HTTPException(status_code=422, detail="instructor must be string")
    if not isinstance(topic, str):
        raise HTTPException(status_code=422, detail="topic must be string")
    if not isinstance(synopsis, str):
        raise HTTPException(status_code=422, detail="synopsis must be string")
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise HTTPException(status_code=422, detail="tags must be list[str]")
    if year is not None:
        # Accept ints, and numeric strings that parse to int.
        if isinstance(year, bool) or not isinstance(year, int):
            raise HTTPException(status_code=422, detail="year must be integer or null")
    if not isinstance(voice_profile, str):
        raise HTTPException(status_code=422, detail="voice_profile must be string")

    return {
        "instructor": instructor,
        "topic": topic,
        "tags": tags,
        "synopsis": synopsis,
        "year": year,
        "voice_profile": voice_profile,
    }


@router.get("/{name}/metadata")
async def get_metadata(name: str):
    import copy
    target = _resolve_target(name)
    sidecar = target / SIDECAR_NAME
    # deepcopy → ``DEFAULT_METADATA["tags"]`` es una list compartida; si el
    # cliente la muta (o se cachea) alterábamos el default global. Misma razón
    # que en settings.load_settings.
    if not sidecar.exists():
        return JSONResponse(copy.deepcopy(DEFAULT_METADATA))
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return JSONResponse(copy.deepcopy(DEFAULT_METADATA))

    # Merge with defaults to guarantee stable shape.
    result = copy.deepcopy(DEFAULT_METADATA)
    if isinstance(raw, dict):
        for k in DEFAULT_METADATA:
            if k in raw:
                result[k] = raw[k]
    # Cache oracle.poster_url alongside so UI can surface re-download state
    # without a second read. Not validated in PUT — oracle owns writes.
    return JSONResponse(result)


@router.put("/{name}/metadata")
async def put_metadata(name: str, request: Request):
    target = _resolve_target(name)
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid JSON")

    payload = _validate_payload(body)

    sidecar = target / SIDECAR_NAME
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
