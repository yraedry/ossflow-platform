"""Dependencias FastAPI del módulo telegram.

Inyecta ``TelegramService`` con la URL del backend (vía la factory
``ossflow_api.clients.telegram.telegram_client``), un loader de
settings (para ``author_aliases``) y un registrar de jobs en background
(para que sync/download aparezcan en el dashboard del processor-api).

Imports diferidos: ``api.settings`` y ``api.background_jobs`` se
importan dentro de la factory para no acoplar este módulo a ellos en
tiempo de import (mismo patrón que el resto de módulos migrados a
vertical slice).
"""

from __future__ import annotations

import logging
from typing import Any

from ossflow_api.clients.telegram import telegram_client

from .service import TelegramService

log = logging.getLogger(__name__)


def _default_settings_loader() -> dict[str, Any]:
    """Carga settings desde la BD para resolver ``author_aliases``."""
    from api.settings import load_settings

    try:
        data = load_settings()
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _build_default_registrar(service: TelegramService):
    """Crea el registrar por defecto cerrando sobre ``service._track_job``.

    El registry de processor-api espera una *coroutine factory* que
    reciba un callback ``update_progress(progress, message)`` y devuelva
    un dict con el resultado. Aquí construimos esa factory enrollando la
    suscripción SSE del backend telegram-fetcher.
    """

    def registrar(kind: str, job_id: str, sse_path: str, params: dict) -> None:
        try:
            from api.background_jobs import registry  # import diferido
        except Exception:  # noqa: BLE001
            log.debug("background_jobs registry unavailable; skipping tracker")
            return

        sse_url = f"{service._backend_base}{sse_path}"  # noqa: SLF001

        async def factory(update_progress):
            update_progress(0.0, f"waiting for {kind}")
            payload = await service._track_job(  # noqa: SLF001
                sse_url=sse_url, job_kind=kind
            )
            evt_type = payload.get("type")
            if evt_type == "error" or payload.get("status") in ("failed",):
                raise RuntimeError(
                    payload.get("data", {}).get("message") or "failed"
                )
            update_progress(100.0, f"{kind} done")
            return {
                "backend_job_id": job_id,
                "last_event": payload,
            }

        enriched = dict(params or {})
        enriched["backend_job_id"] = job_id
        enriched["sse_url"] = sse_path
        try:
            registry.submit(kind, factory, enriched)
        except Exception:  # noqa: BLE001
            log.exception("failed to register %s background job", kind)

    return registrar


def get_telegram_service() -> TelegramService:
    """Factory inyectada vía ``Depends()`` por el router."""
    client = telegram_client()
    # Construimos el servicio sin registrar primero para poder cerrar el
    # registrar sobre la propia instancia (lo necesita para acceder al
    # ``_backend_base`` y al método ``_track_job`` de la subscripción SSE).
    service = TelegramService(
        base_url=client.base_url,
        settings_loader=_default_settings_loader,
        bg_job_registrar=None,
    )
    service._register_bg_job = _build_default_registrar(service)  # noqa: SLF001
    return service
