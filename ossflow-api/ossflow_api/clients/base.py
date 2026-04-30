"""HTTP client base para microservicios backend (splitter, subs, dubbing).

Single responsibility: hablar HTTP + parsear SSE. Nada más.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

from ossflow_api.shared.events import NormalizedEvent, is_terminal, normalize

log = logging.getLogger(__name__)

RUN_TIMEOUT = 10.0
STREAM_RECONNECT_DELAY = 2.0
# SSE streams: el backend envía heartbeat cada ~15 s. Si pasan 120 s
# sin ningún dato el backend está colgado → reconectamos. connect/write
# son rápidos (<10 s). pool es el slot del connection pool; alto para
# no bloquear.
_STREAM_TIMEOUT = httpx.Timeout(
    connect=10.0, read=120.0, write=10.0, pool=30.0,
)


class BackendError(RuntimeError):
    """Lanzada cuando un backend devuelve respuesta de error."""


class BackendClient:
    """Cliente HTTP async para un microservicio backend."""

    def __init__(self, base_url: str, *, run_timeout: float = RUN_TIMEOUT) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self._run_timeout = run_timeout

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._run_timeout) as client:
            r = await client.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()

    async def run(self, payload: dict[str, Any]) -> str:
        """POST /run con payload, devuelve job_id de la respuesta."""
        return await self._post_job("/run", payload)

    async def run_oracle(self, payload: dict[str, Any]) -> str:
        """POST /run-oracle con payload, devuelve job_id de la respuesta."""
        return await self._post_job("/run-oracle", payload)

    async def _post_job(self, path: str, payload: dict[str, Any]) -> str:
        async with httpx.AsyncClient(timeout=self._run_timeout) as client:
            r = await client.post(f"{self.base_url}{path}", json=payload)
            if r.status_code >= 400:
                raise BackendError(f"{r.status_code}: {r.text}")
            data = r.json()
            job_id = data.get("job_id") or data.get("id")
            if not job_id:
                raise BackendError(f"No job_id in response: {data}")
            return job_id

    async def stream(
        self, job_id: str, *, max_reconnects: int = 3
    ) -> AsyncIterator[NormalizedEvent]:
        """Stream SSE events de /events/{job_id}, yield NormalizedEvent.

        Acepta tanto el contrato de ``ossflow_service_kit`` (``{"type","data"}``)
        como el contrato flat legacy (``{"status","progress",...}``).
        Reconecta en disconnect transitorios hasta ``max_reconnects`` veces.
        Termina cuando llega un evento terminal (done/error).

        Manejo de 404: si una reconexión recibe 404 *después* de que ya hayamos
        visto al menos un evento, el job casi seguro completó y se reapeó del
        registry en memoria del backend entre nuestros intentos de reconexión
        (o el backend reinició para liberar VRAM). Tratamos eso como "stream
        cerrado limpiamente" en vez de error backend — la alternativa sería
        marcar un step exitoso como FAILED, que observamos en jobs de doblaje
        largos (~70 min) donde las transiciones síntesis→mezcla→mux pueden
        estar silenciosas >120 s y disparar el read timeout.

        404 en el primer intento (sin eventos vistos aún) sigue lanzando — eso
        es un "job_id no encontrado" genuino y probablemente bug del caller.
        """
        url = f"{self.base_url}/events/{job_id}"
        attempts = 0
        seen_any_event = False
        while True:
            try:
                async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
                    async with client.stream("GET", url) as resp:
                        if resp.status_code == 404 and seen_any_event:
                            log.info(
                                "SSE 404 on reconnect for %s — job likely "
                                "completed and reaped, treating as clean close",
                                url,
                            )
                            return
                        if resp.status_code >= 400:
                            raise BackendError(
                                f"stream {resp.status_code} on {url}"
                            )
                        buffer: list[str] = []
                        async for line in resp.aiter_lines():
                            if line == "":
                                if buffer:
                                    raw = _parse_sse_block(buffer)
                                    buffer = []
                                    if raw is not None:
                                        evt = normalize(raw)
                                        seen_any_event = True
                                        yield evt
                                        if is_terminal(evt):
                                            return
                                continue
                            buffer.append(line)
                        # stream closed cleanly
                        return
            except (
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ) as exc:
                attempts += 1
                if attempts > max_reconnects:
                    raise BackendError(f"SSE reconnect limit reached: {exc}") from exc
                log.warning(
                    "SSE disconnect on %s (attempt %d/%d): %s",
                    url, attempts, max_reconnects, exc,
                )
                await asyncio.sleep(STREAM_RECONNECT_DELAY)


def _parse_sse_block(lines: list[str]) -> Optional[dict[str, Any]]:
    """Parsea un bloque SSE (líneas entre líneas en blanco)."""
    data_parts: list[str] = []
    for ln in lines:
        if ln.startswith(":"):
            continue  # comment / heartbeat
        if ln.startswith("data:"):
            data_parts.append(ln[5:].lstrip())
    if not data_parts:
        return None
    raw = "\n".join(data_parts)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
