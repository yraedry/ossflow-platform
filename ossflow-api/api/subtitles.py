"""Proxy endpoints to subtitle-generator for per-segment validation & regeneration."""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.paths import to_container_path
from api.settings import get_library_path, get_setting

log = logging.getLogger(__name__)

SUBS_URL = os.environ.get("SUBS_URL", "http://localhost:8002")

router = APIRouter(prefix="/api/subtitles", tags=["subtitles"])


class ValidateBody(BaseModel):
    srt_path: str


class RegenerateBody(BaseModel):
    srt_path: str
    segment_idx: int
    context_seconds: float = 1.0
    video_path: Optional[str] = None
    model: str = "large-v3"
    language: str = "en"


class ApplyBody(BaseModel):
    srt_path: str
    segment_idx: int
    text: str
    start: Optional[float] = None
    end: Optional[float] = None


class TranslateBody(BaseModel):
    srt_path: str
    target_lang: str = "ES"
    source_lang: str = "EN"
    provider: Optional[str] = None
    model: Optional[str] = None
    formality: Optional[str] = None
    api_key: Optional[str] = None
    fallback_provider: Optional[str] = None
    fallback_api_key: Optional[str] = None
    out_path: Optional[str] = None
    dubbing_mode: bool = False
    dubbing_cps: Optional[float] = None


class AnalyzeBody(BaseModel):
    video_path: str
    language: str = "en"
    model: str = "large-v3"


def _translate(host_path: str) -> str:
    lib = get_library_path()
    if not lib:
        return host_path
    try:
        return to_container_path(host_path, lib)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


async def _post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(f"{SUBS_URL}{path}", json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"subtitle-generator: {exc}")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


@router.post("/validate")
async def validate(body: ValidateBody) -> dict:
    payload = {"srt_path": _translate(body.srt_path)}
    return await _post("/validate", payload, timeout=30.0)


@router.post("/regenerate-segment")
async def regenerate_segment(body: RegenerateBody) -> dict:
    payload = {
        "srt_path": _translate(body.srt_path),
        "segment_idx": body.segment_idx,
        "context_seconds": body.context_seconds,
        "model": body.model,
        "language": body.language,
    }
    if body.video_path:
        payload["video_path"] = _translate(body.video_path)
    return await _post("/regenerate-segment", payload, timeout=300.0)


@router.post("/apply-segment")
async def apply_segment(body: ApplyBody) -> dict:
    payload = {
        "srt_path": _translate(body.srt_path),
        "segment_idx": body.segment_idx,
        "text": body.text,
    }
    if body.start is not None:
        payload["start"] = body.start
    if body.end is not None:
        payload["end"] = body.end
    return await _post("/apply-segment", payload, timeout=30.0)


@router.post("/maintenance/clear-locks")
async def clear_locks() -> dict:
    """Clear stale HuggingFace hub locks on the subtitle-generator side."""
    return await _post("/maintenance/clear-hf-locks", {}, timeout=15.0)


@router.post("/maintenance/restart")
async def restart_subtitle_service() -> dict:
    """Proxy graceful restart to subtitle-generator (releases VRAM, Docker restarts container)."""
    return await _post("/maintenance/restart", {}, timeout=15.0)


def _key_for_provider(provider: str, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    p = (provider or "").lower().strip()
    if p == "openai":
        return get_setting("openai_api_key")
    if p == "ollama":
        return None  # ollama no necesita api_key
    return None


@router.post("/translate")
async def translate(body: TranslateBody) -> dict:
    provider = (body.provider or get_setting("translation_provider") or "ollama").lower()
    fallback = (
        body.fallback_provider
        or get_setting("translation_fallback_provider")
        or ""
    ).lower() or None
    model = body.model or get_setting("translation_model")

    payload: dict = {
        "srt_path": _translate(body.srt_path),
        "target_lang": body.target_lang,
        "source_lang": body.source_lang,
        "provider": provider,
    }
    if model:
        payload["model"] = model
    if body.formality:
        payload["formality"] = body.formality

    api_key = _key_for_provider(provider, body.api_key)
    if provider != "ollama" and not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"{provider} API key missing (set it in Settings)",
        )
    if api_key:
        payload["api_key"] = api_key

    if fallback and fallback != provider:
        fb_key = _key_for_provider(fallback, body.fallback_api_key)
        if fallback == "ollama" or fb_key:
            payload["fallback_provider"] = fallback
            if fb_key:
                payload["fallback_api_key"] = fb_key

    if body.out_path:
        payload["out_path"] = _translate(body.out_path)
    if body.dubbing_mode:
        payload["dubbing_mode"] = True
        cps = body.dubbing_cps or get_setting("translation_dubbing_cps")
        if cps:
            payload["dubbing_cps"] = float(cps)
    return await _post("/translate", payload, timeout=600.0)


@router.post("/analyze")
async def analyze_video(body: AnalyzeBody) -> dict:
    payload = {
        "video_path": _translate(body.video_path),
        "language": body.language,
        "model": body.model,
    }
    return await _post("/analyze", payload, timeout=600.0)
