"""Endpoints HTTP del módulo dubbing.

Reproduce el contrato del antiguo ``api/dubbing.py``: mismos paths,
mismos métodos, mismos payloads. La lógica vive en ``DubbingService``;
este router sólo valida con Pydantic y delega.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ossflow_api.shared.exceptions import ApiError

from .dependencies import get_dubbing_service
from .schemas import AnalyzeBody, VoiceTranscriptBody
from .service import DubbingService

router = APIRouter(prefix="/api/dubbing", tags=["dubbing"])


def _to_http(exc: ApiError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@router.get("/voices")
async def list_voices(
    svc: DubbingService = Depends(get_dubbing_service),
) -> dict:
    """Lista los WAV de referencia (voces ES) disponibles dentro del backend."""
    try:
        return await svc.list_voices()
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.put("/voices/{filename}/transcript")
async def save_voice_transcript(
    filename: str,
    body: VoiceTranscriptBody,
    svc: DubbingService = Depends(get_dubbing_service),
) -> dict:
    """Persiste una transcripción de referencia como sidecar junto al WAV."""
    try:
        return await svc.save_voice_transcript(filename, body)
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.get("/qa")
def get_dub_qa(
    video_path: str,
    svc: DubbingService = Depends(get_dubbing_service),
) -> dict:
    """Devuelve el sidecar ``dub-qa.json`` generado por el dubbing pipeline."""
    try:
        return svc.get_dub_qa(video_path)
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.get("/qa/instructional/{name}")
def get_instructional_qa(
    name: str,
    svc: DubbingService = Depends(get_dubbing_service),
) -> dict:
    """QA agregado para todos los capítulos doblados de un instruccional."""
    try:
        return svc.get_instructional_qa(name)
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.post("/maintenance/restart")
async def restart_dubbing_service(
    svc: DubbingService = Depends(get_dubbing_service),
) -> dict:
    """Proxy del restart graceful (libera VRAM; Docker reinicia el contenedor)."""
    try:
        return await svc.restart()
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.post("/analyze")
async def analyze_dubbing(
    body: AnalyzeBody,
    svc: DubbingService = Depends(get_dubbing_service),
) -> dict:
    try:
        return await svc.analyze(body)
    except ApiError as exc:
        raise _to_http(exc) from exc
