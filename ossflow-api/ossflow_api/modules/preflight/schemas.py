"""DTOs y constantes del módulo preflight."""

from __future__ import annotations

from dataclasses import dataclass


MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
HEALTH_TIMEOUT = 2.0
CACHE_TTL_SECONDS = 30.0
STATIC_CACHE_TTL_SECONDS = 5 * 60.0  # 5 min


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
