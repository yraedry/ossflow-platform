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
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from . import filesystem as _fs
from . import media as _media
from . import mount as _mount
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


# ---------------------------------------------------------------------------
# Filesystem browsing (T23.4)
# ---------------------------------------------------------------------------


@router.get("/api/fs/browse")
async def api_fs_browse(path: str = ""):
    """Lista subdirectorios bajo ``MEDIA_ROOT`` (anti-traversal)."""
    try:
        return _fs.fs_browse(path)
    except _fs._NoMediaRoot as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except _fs._OutOfRoot as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except _fs._NotADir as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except _fs._NoPermission as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)


@router.get("/api/browse")
async def api_browse(
    path: Optional[str] = None,
    svc: LibraryService = Depends(get_library_service),
):
    """Browse libre desde ``library_path`` o ``MEDIA_ROOT``."""
    try:
        return _fs.browse(path, svc._library_path_loader)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except _fs._NotADir as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except _fs._NoPermission as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    except OSError as exc:
        return JSONResponse({"error": f"Error al leer directorio: {exc}"}, status_code=500)


# ---------------------------------------------------------------------------
# NAS mount (T23.4)
# ---------------------------------------------------------------------------


@router.post("/api/mount")
async def api_mount(body: dict):
    try:
        return _mount.mount_share(body)
    except _mount._BadRequest as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except _mount._MountFailed as exc:
        return JSONResponse(
            {
                "error": f"No se pudo montar: {exc}",
                "hint": "Verifica la IP, la ruta compartida y las credenciales.",
            },
            status_code=500,
        )


@router.get("/api/mount")
async def api_mount_status():
    return _mount.mount_status()


# ---------------------------------------------------------------------------
# Media (T23.5): video-info, thumbnail, streaming Range-aware
# ---------------------------------------------------------------------------


@router.get("/api/video-info")
async def api_video_info(path: str):
    if not Path(path).exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return _media.video_info(path)


@router.get("/api/thumbnail")
async def api_thumbnail(
    path: str,
    t: float = 5.0,
    svc: LibraryService = Depends(get_library_service),
):
    """Genera thumbnail con traducción host→container del path.

    El backend corre en un container que monta ``library_path`` como
    ``/library``; ``to_container_path`` mapea el path absoluto del NAS
    a la ruta visible para ffmpeg.
    """
    import os as _os

    from api.paths import to_container_path

    try:
        lib = svc._library_path_loader() or ""
        container_root = _os.environ.get("MEDIA_ROOT", "/media")
        container_path = (
            to_container_path(path, lib, container_root) if lib else path
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not Path(container_path).exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    thumb = _media.generate_thumbnail(container_path, t)
    if thumb:
        return StreamingResponse(
            iter([thumb]),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    return JSONResponse({"error": "Could not generate thumbnail"}, status_code=500)


@router.get("/api/media")
async def api_media(path: str, request: Request):
    """Sirve un fichero media (vídeo/subtitle) con HTTP Range para seek."""
    target = _media.resolve_media_path(path)
    if target is None:
        return JSONResponse(
            {"error": "not found or outside MEDIA_ROOT"}, status_code=404,
        )

    ext = target.suffix.lower()
    mime = _media.MEDIA_MIME.get(ext, "application/octet-stream")
    size = target.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range")

    # Subtitles: serve whole file, convert SRT → VTT on the fly when asked.
    if ext in (".srt", ".vtt"):
        if ext == ".srt" and request.query_params.get("as") == "vtt":
            raw = target.read_text(encoding="utf-8", errors="replace")
            vtt = "WEBVTT\n\n" + raw.replace(",", ".")
            return Response(content=vtt, media_type="text/vtt")
        return FileResponse(path=str(target), media_type=mime)

    # No Range → full file.
    if not range_header:
        return FileResponse(
            path=str(target),
            media_type=mime,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(size)},
        )

    try:
        units, _, rng = range_header.partition("=")
        if units.strip().lower() != "bytes":
            raise ValueError
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
        if start < 0 or end >= size or start > end:
            raise ValueError
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    chunk_size = 1024 * 1024
    length = end - start + 1

    def _iter():
        with open(target, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                data = f.read(min(chunk_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        _iter(),
        status_code=206,
        media_type=mime,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )
