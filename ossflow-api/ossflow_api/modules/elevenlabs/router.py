"""Endpoints HTTP del módulo elevenlabs."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .service import already_dubbed, get_job, list_jobs, spawn_job

router = APIRouter(prefix="/api/elevenlabs-dubbing", tags=["elevenlabs-dubbing"])

_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}


@router.post("")
async def create_elevenlabs_dubbing_job(request: Request) -> dict:
    """Encola un job ElevenLabs Dubbing Studio para un único vídeo.

    Body::

        {
          "path": "/media/.../S01E02 - Foo.mp4",  # REQUIRED
          "source_lang": "en",                     # default "en"
          "target_lang": "es",                     # default "es"
          "num_speakers": 1,                       # default 1
          "watermark": true                        # default true
        }
    """
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    path = body.get("path")
    if not path or not isinstance(path, str):
        raise HTTPException(status_code=400, detail="'path' is required")
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {path}")

    source_lang = str(body.get("source_lang") or "en").strip() or "en"
    target_lang = str(body.get("target_lang") or "es").strip() or "es"
    num_speakers = int(body.get("num_speakers") or 1)
    watermark = bool(body.get("watermark", True))

    job_id = spawn_job(
        source,
        source_lang=source_lang,
        target_lang=target_lang,
        num_speakers=num_speakers,
        watermark=watermark,
    )
    return {"job_id": job_id, "status": "queued"}


@router.post("/batch")
async def create_elevenlabs_dubbing_batch(request: Request) -> dict:
    """Encola jobs ElevenLabs para todos los vídeos de una temporada.

    Body::

        {
          "season_path": "/media/.../Season 01",  # REQUIRED
          "source_lang": "en",
          "target_lang": "es",
          "num_speakers": 1,
          "watermark": true
        }

    Vídeos ya doblados (``<season>/elevenlabs/<filename>`` con tamaño
    plausible) se saltan para no quemar créditos al relanzar un batch.
    """
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    season_path = body.get("season_path")
    if not season_path or not isinstance(season_path, str):
        raise HTTPException(status_code=400, detail="'season_path' is required")
    season = Path(season_path)
    if not season.exists() or not season.is_dir():
        raise HTTPException(status_code=404, detail=f"Season folder not found: {season_path}")

    source_lang = str(body.get("source_lang") or "en").strip() or "en"
    target_lang = str(body.get("target_lang") or "es").strip() or "es"
    num_speakers = int(body.get("num_speakers") or 1)
    watermark = bool(body.get("watermark", True))

    queued: list[dict] = []
    skipped: list[dict] = []
    for child in sorted(season.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_file() or child.suffix.lower() not in _VIDEO_EXTS:
            continue
        if child.stem.endswith("_DOBLADO"):
            continue
        if already_dubbed(child):
            skipped.append({"filename": child.name, "reason": "already_dubbed"})
            continue
        job_id = spawn_job(
            child,
            source_lang=source_lang,
            target_lang=target_lang,
            num_speakers=num_speakers,
            watermark=watermark,
        )
        queued.append({"filename": child.name, "job_id": job_id})

    return {
        "season_path": str(season),
        "queued": queued,
        "skipped": skipped,
        "queued_count": len(queued),
        "skipped_count": len(skipped),
    }


@router.get("")
async def list_elevenlabs_dubbing_jobs(limit: int = 50) -> dict:
    """Lista jobs ElevenLabs (activos + últimos ``limit`` completados)."""
    return list_jobs(limit=limit)


@router.get("/{job_id}")
async def get_elevenlabs_dubbing_status(job_id: str):
    """Alias provider-scoped sobre el registry global de jobs."""
    job = get_job(job_id)
    if job is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return job
