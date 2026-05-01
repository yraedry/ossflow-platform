"""Módulo logs: agrega ring buffer local + proxy a /logs de los backends."""

from .router import router as logs_router
from .service import RingBufferHandler, install_local_ring_buffer

__all__ = ["logs_router", "RingBufferHandler", "install_local_ring_buffer"]
