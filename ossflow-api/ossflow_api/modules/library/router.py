"""Router del módulo library.

Endpoints HTTP migrados literalmente de ``api/app.py``. Mantiene los
mismos paths, status codes y shape de respuesta que el frontend espera.

Endpoints registrados:

* ``POST /api/scan`` — escaneo completo + persistencia.
* ``GET  /api/library`` — cache + opcional rescan en background.
* ``GET  /api/library/{name}`` — detalle agrupado por season.
* ``POST /api/library/{name}/refresh`` — re-discover de un instructional.
* ``GET  /api/library/{name}/poster`` — sirve poster con ETag/304.
* ``POST /api/library/{name}/poster`` — upload custom poster.
* ``POST /api/library/{name}/poster/redownload`` — re-baja del scrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from .dependencies import get_library_service
from .schemas import ScanRequest
from .service import LibraryService, _Forbidden, _NotFound

router = APIRouter(tags=["library"])


_ALLOWED_POSTER_EXT = {"jpg", "jpeg", "png", "webp"}
_POSTER_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


@router.post("/api/scan")
async def api_scan(
    request: Request,
    svc: LibraryService = Depends(get_library_service),
):
    """Lanza un escaneo completo de la biblioteca.

    Si ``path`` viene vacío en el body, usa ``settings.library_path``.
    """
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}
    root_path = body.get("path", "") or ""

    try:
        return await svc.scan(root_path=root_path or None)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except _NotFound as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)


@router.get("/api/library")
async def api_library(
    refresh: bool = False,
    svc: LibraryService = Depends(get_library_service),
):
    """Devuelve la biblioteca cacheada (con refresh opcional en background)."""
    return svc.get_cached(refresh=refresh)


@router.get("/api/library/{name}")
async def api_library_detail(
    name: str,
    refresh: bool = True,
    svc: LibraryService = Depends(get_library_service),
):
    try:
        return await svc.get_detail(name, refresh=refresh)
    except _NotFound as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)


@router.post("/api/library/{name}/refresh")
async def api_library_refresh(
    name: str,
    svc: LibraryService = Depends(get_library_service),
):
    try:
        return await svc.refresh_instructional(name)
    except _NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/api/library/{name}/poster")
async def api_library_poster(
    name: str,
    request: Request,
    svc: LibraryService = Depends(get_library_service),
):
    """Sirve el poster con ETag estable basado en mtime+size + 304."""
    try:
        _target, poster = svc.find_cached_poster(name)
    except _Forbidden as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    except _NotFound as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    try:
        st = poster.stat()
        etag = f'"{st.st_mtime_ns}-{st.st_size}"'
    except OSError:
        etag = None

    cache_control = "public, max-age=86400, stale-while-revalidate=604800"

    if etag is not None:
        inm = request.headers.get("if-none-match")
        if inm and etag in [v.strip() for v in inm.split(",")]:
            return Response(
                status_code=304,
                headers={"ETag": etag, "Cache-Control": cache_control},
            )

    ext = poster.suffix.lower().lstrip(".")
    media_type = _POSTER_MEDIA_TYPES.get(ext, "application/octet-stream")
    headers = {"Cache-Control": cache_control}
    if etag is not None:
        headers["ETag"] = etag
    return FileResponse(path=str(poster), media_type=media_type, headers=headers)


@router.post("/api/library/{name}/poster")
async def api_library_poster_upload(
    name: str,
    file: UploadFile = File(...),
    svc: LibraryService = Depends(get_library_service),
):
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in _ALLOWED_POSTER_EXT:
        raise HTTPException(status_code=415, detail=f"unsupported extension: {ext}")

    contents = await file.read()
    try:
        return svc.upload_poster(name, ext, contents)
    except _Forbidden as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except _NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        msg = str(exc)
        status = 413 if "too large" in msg else 415
        raise HTTPException(status_code=status, detail=msg)


@router.post("/api/library/{name}/poster/redownload")
async def api_library_poster_redownload(
    name: str,
    svc: LibraryService = Depends(get_library_service),
):
    try:
        return await svc.redownload_poster(name)
    except _Forbidden as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except _NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        msg = str(exc)
        status = 502 if "download failed" in msg else 500
        raise HTTPException(status_code=status, detail=msg)
