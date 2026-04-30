"""Servicio de telegram: proxy fino hacia el backend telegram-fetcher.

Responsabilidades:

* Hacer ``GET/POST/PUT/PATCH/DELETE {base_url}/telegram{path}`` con
  timeouts y mapear errores ``httpx`` a ``HTTPException`` (502 / 504),
  preservando los códigos 4xx/5xx que devuelve el backend.
* Proxyar streams SSE (``/events``) vía ``StreamingResponse`` chunked.
* Registrar entradas en el ``background_jobs`` registry del processor-api
  cuando el backend acepta un sync o download (para que aparezcan en
  el dashboard junto a cleanups / duplicate-scans).
* Aplicar ``author_aliases`` de settings al listado por autor para
  fusionar nombres equivalentes antes de devolver el payload.

Los nombres de paths backend son los del antiguo ``api/telegram.py``
(``/status``, ``/auth/...``, ``/channels...``, ``/media...``,
``/download...``, ``/syncs/active``). Se preserva el comportamiento exacto
del proxy legacy; sólo cambia el empaquetado para encajar en el patrón
vertical slice y permitir inyección de dependencias en tests.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .schemas import DEFAULT_TIMEOUT, SSE_TIMEOUT

log = logging.getLogger(__name__)


# Tipo de la función opcional que registra un job en el registry del dashboard.
# Recibe (kind, job_id, sse_path, params) y no devuelve nada.
BgJobRegistrar = Callable[[str, str, str, dict], None]

# Tipo del loader de settings (usado para author_aliases). Debe devolver un dict.
SettingsLoader = Callable[[], dict]


class TelegramService:
    """Cliente de proxy hacia telegram-fetcher.

    ``base_url`` apunta al backend SIN sufijo ``/telegram`` — el
    servicio compone ``{base_url}/telegram{path}`` en cada request,
    porque los routers del microservicio están montados bajo ese
    prefijo.

    ``settings_loader`` y ``bg_job_registrar`` se inyectan para que los
    tests puedan reemplazarlos sin tocar la BD ni el registry global.
    Cualquiera de ellos puede ser ``None`` para deshabilitar el
    feature: en ese caso ``_apply_author_aliases`` se vuelve no-op y
    ``_register_bg_job`` se ignora silenciosamente.
    """

    def __init__(
        self,
        *,
        base_url: str,
        settings_loader: Optional[SettingsLoader] = None,
        bg_job_registrar: Optional[BgJobRegistrar] = None,
    ) -> None:
        # Mismo cálculo de URL que el legacy: añade ``/telegram`` una sola vez.
        self._backend_base = f"{base_url.rstrip('/')}/telegram"
        self._settings_loader = settings_loader
        self._register_bg_job = bg_job_registrar

    # ------------------------------------------------------------------
    # Helpers HTTP
    # ------------------------------------------------------------------

    async def _proxy_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
    ) -> JSONResponse:
        """Reenvía una petición JSON al backend y devuelve ``JSONResponse``.

        - Timeout backend → 504.
        - Cualquier otro fallo httpx → 502.
        - 4xx/5xx del backend se reenvían tal cual (status + body).
        """
        url = f"{self._backend_base}{path}"
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                r = await client.request(method, url, json=json_body, params=params)
        except httpx.TimeoutException as exc:
            log.warning("telegram backend timeout %s %s: %s", method, path, exc)
            raise HTTPException(status_code=504, detail=f"backend timeout: {exc}")
        except httpx.HTTPError as exc:
            log.warning("telegram backend unreachable %s %s: %s", method, path, exc)
            raise HTTPException(status_code=502, detail=f"backend unreachable: {exc}")

        try:
            payload = r.json()
        except ValueError:
            payload = {"detail": r.text}

        return JSONResponse(payload, status_code=r.status_code)

    async def _sse_proxy(self, path: str) -> StreamingResponse:
        """Reenvía un stream SSE del backend al cliente.

        El status del envoltorio siempre es 200; los errores aguas
        arriba se materializan como ``event: error`` en el flujo, igual
        que el proxy legacy.
        """
        url = f"{self._backend_base}{path}"

        async def gen() -> AsyncIterator[bytes]:
            try:
                async with httpx.AsyncClient(timeout=SSE_TIMEOUT) as client:
                    async with client.stream("GET", url) as r:
                        if r.status_code >= 400:
                            body = await r.aread()
                            yield (
                                f"event: error\ndata: backend {r.status_code}: "
                                f"{body.decode('utf-8', errors='replace')}\n\n"
                            ).encode("utf-8")
                            return
                        async for chunk in r.aiter_bytes():
                            if chunk:
                                yield chunk
            except httpx.HTTPError as exc:
                yield (
                    f"event: error\ndata: backend unreachable: {exc}\n\n"
                ).encode("utf-8")

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Tracking de jobs en background (dashboard "Jobs activos")
    # ------------------------------------------------------------------

    async def _track_job(self, *, sse_url: str, job_kind: str) -> dict:
        """Suscribe al SSE del backend y devuelve el último evento o
        el evento terminal (``done`` / ``error`` / ``cancelled``).
        """
        last_event: dict = {}
        try:
            async with httpx.AsyncClient(timeout=SSE_TIMEOUT) as client:
                async with client.stream("GET", sse_url) as r:
                    if r.status_code >= 400:
                        raise RuntimeError(f"{job_kind} sse error: {r.status_code}")
                    buf = b""
                    async for chunk in r.aiter_bytes():
                        if not chunk:
                            continue
                        buf += chunk
                        while b"\n\n" in buf:
                            raw, buf = buf.split(b"\n\n", 1)
                            event_type = "message"
                            data_txt = ""
                            for line in raw.splitlines():
                                if line.startswith(b"event:"):
                                    event_type = line.split(b":", 1)[1].strip().decode()
                                elif line.startswith(b"data:"):
                                    data_txt += line.split(b":", 1)[1].strip().decode()
                            if not data_txt:
                                continue
                            try:
                                payload = json.loads(data_txt)
                            except ValueError:
                                continue
                            last_event = payload
                            if event_type in ("done", "error", "cancelled"):
                                return payload
        except httpx.HTTPError as exc:
            log.warning("telegram tracker: backend unreachable: %s", exc)
        return last_event

    def _maybe_register_bg_job(
        self,
        kind: str,
        job_id: str,
        sse_path: str,
        params: dict,
    ) -> None:
        """Registra un job en el registry sólo si el registrar fue inyectado."""
        if self._register_bg_job is None:
            return
        try:
            self._register_bg_job(kind, job_id, sse_path, params)
        except Exception:  # noqa: BLE001
            log.exception("failed to register %s background job", kind)

    # ------------------------------------------------------------------
    # Status + Auth
    # ------------------------------------------------------------------

    async def get_status(self) -> JSONResponse:
        return await self._proxy_json("GET", "/status")

    async def auth_send_code(self, phone: str) -> JSONResponse:
        return await self._proxy_json(
            "POST", "/auth/send-code", json_body={"phone": phone}
        )

    async def auth_sign_in(self, payload: dict[str, Any]) -> JSONResponse:
        return await self._proxy_json("POST", "/auth/sign-in", json_body=payload)

    async def auth_2fa(self, password: str) -> JSONResponse:
        return await self._proxy_json(
            "POST", "/auth/2fa", json_body={"password": password}
        )

    async def auth_logout(self) -> JSONResponse:
        return await self._proxy_json("POST", "/auth/logout")

    # ------------------------------------------------------------------
    # Channels + Sync
    # ------------------------------------------------------------------

    async def list_channels(self) -> JSONResponse:
        return await self._proxy_json("GET", "/channels")

    async def add_channel(self, username: str) -> JSONResponse:
        return await self._proxy_json(
            "POST", "/channels", json_body={"username": username}
        )

    async def update_channel(self, channel_id: str, title: str) -> JSONResponse:
        return await self._proxy_json(
            "PATCH", f"/channels/{channel_id}", json_body={"title": title}
        )

    async def delete_channel(self, channel_id: str) -> JSONResponse:
        return await self._proxy_json("DELETE", f"/channels/{channel_id}")

    async def list_active_syncs(self) -> JSONResponse:
        return await self._proxy_json("GET", "/syncs/active")

    async def sync_channel(
        self, username: str, payload: dict[str, Any]
    ) -> JSONResponse:
        resp = await self._proxy_json(
            "POST", f"/channels/{username}/sync", json_body=payload
        )
        # Registra dashboard job en respuesta exitosa.
        try:
            if 200 <= resp.status_code < 300:
                body_bytes = resp.body if hasattr(resp, "body") else b""
                data = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                job_id = data.get("job_id") or data.get("id")
                if job_id:
                    self._maybe_register_bg_job(
                        "telegram_sync",
                        job_id,
                        f"/channels/{username}/sync/{job_id}/events",
                        {"username": username, "channel": username},
                    )
        except Exception:  # noqa: BLE001
            log.exception("failed to track telegram_sync")
        return resp

    async def sync_channel_events(
        self, username: str, job_id: str
    ) -> StreamingResponse:
        return await self._sse_proxy(f"/channels/{username}/sync/{job_id}/events")

    # ------------------------------------------------------------------
    # Media
    # ------------------------------------------------------------------

    async def list_media(self, params: dict[str, Any]) -> JSONResponse | dict:
        """Listado de media. Para vista por autor, mezcla ``author_aliases``."""
        view = params.get("view")
        data = await self._proxy_json("GET", "/media", params=params)
        if view == "by_author" and isinstance(data, dict):
            return self._apply_author_aliases(data)
        return data

    def _apply_author_aliases(self, data: dict) -> dict:
        """Fusiona buckets de autor según ``author_aliases`` de settings.

        ``author_aliases`` mapea ``"raw name"`` → ``"canonical name"``;
        los grupos cuyo autor coincida (case-insensitive) con una clave
        se reescriben y se concatenan en el bucket canónico.
        """
        if self._settings_loader is None:
            return data
        try:
            aliases_raw = self._settings_loader().get("author_aliases") or {}
        except Exception:  # noqa: BLE001
            return data
        if not aliases_raw:
            return data

        lookup = {
            str(k).strip().lower(): str(v).strip()
            for k, v in aliases_raw.items()
            if k and v
        }
        if not lookup:
            return data

        authors = data.get("authors")
        if not isinstance(authors, list):
            return data

        merged: dict[str, dict] = {}
        order: list[str] = []
        for a in authors:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name") or "").strip()
            canonical = lookup.get(name.lower(), name)
            bucket = merged.get(canonical)
            if bucket is None:
                bucket = {
                    **a,
                    "name": canonical,
                    "instructionals": list(a.get("instructionals") or []),
                }
                merged[canonical] = bucket
                order.append(canonical)
            else:
                bucket["instructionals"].extend(a.get("instructionals") or [])

        out_authors = [merged[k] for k in order]
        out_authors.sort(key=lambda x: str(x.get("name") or "").lower())
        return {**data, "authors": out_authors}

    async def get_media_thumbnail(
        self, channel_id: str, message_id: str
    ) -> Response:
        """Reenvía el thumbnail binario del backend.

        Stream directo de bytes; sin buffering por card. 404 propaga;
        cualquier 4xx/5xx se mapea a ``HTTPException``.
        """
        url = (
            f"{self._backend_base}/media/{channel_id}/{message_id}/thumbnail"
        )
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                r = await client.get(url)
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail=f"backend timeout: {exc}")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"backend unreachable: {exc}")
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="thumbnail not available")
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return Response(
            content=r.content,
            media_type=r.headers.get("content-type", "image/jpeg"),
            headers={"Cache-Control": "public, max-age=86400"},
        )

    async def put_media_metadata(
        self,
        channel_id: str,
        message_id: str,
        payload: dict[str, Any],
    ) -> JSONResponse:
        return await self._proxy_json(
            "PUT", f"/media/{channel_id}/{message_id}", json_body=payload
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    async def start_download(self, payload: dict[str, Any]) -> JSONResponse:
        resp = await self._proxy_json("POST", "/download", json_body=payload)
        try:
            if 200 <= resp.status_code < 300:
                body_bytes = resp.body if hasattr(resp, "body") else b""
                data = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                job_id = data.get("job_id") or data.get("id")
                if job_id:
                    self._maybe_register_bg_job(
                        "telegram_download",
                        job_id,
                        f"/download/{job_id}/events",
                        {
                            "author": payload.get("author"),
                            "title": payload.get("title"),
                            "channel_id": payload.get("channel_id"),
                        },
                    )
        except Exception:  # noqa: BLE001
            log.exception("failed to track telegram_download")
        return resp

    async def download_events(self, job_id: str) -> StreamingResponse:
        return await self._sse_proxy(f"/download/{job_id}/events")

    async def cancel_download(self, job_id: str) -> JSONResponse:
        return await self._proxy_json("POST", f"/download/{job_id}/cancel")

    async def list_download_jobs(self, status: str | None) -> JSONResponse:
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        return await self._proxy_json("GET", "/download/jobs", params=params)
