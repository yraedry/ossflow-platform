"""In-memory auth state + lightweight JSON metadata on disk.

Policy:
- ``phone_code_hash`` lives ONLY in memory (ephemeral by Telegram spec).
- ``phone`` and ``last_login_iso`` are persisted in ``/data/session/auth_meta.json``.
- State transitions are validated to avoid nonsensical flows.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Optional

from .models import AuthState, AuthStateLiteral


DEFAULT_META_PATH = "/data/session/auth_meta.json"


_VALID_TRANSITIONS: dict[AuthStateLiteral, set[AuthStateLiteral]] = {
    "disconnected": {"awaiting_code", "authenticated", "disconnected"},
    "awaiting_code": {"awaiting_2fa", "authenticated", "disconnected", "awaiting_code"},
    "awaiting_2fa": {"authenticated", "disconnected", "awaiting_2fa"},
    "authenticated": {"disconnected", "authenticated"},
}


def _resolve_meta_path(path: Optional[str]) -> str:
    return path or os.environ.get("TG_AUTH_META") or DEFAULT_META_PATH


class InvalidTransition(RuntimeError):
    pass


class AuthStateStore:
    """Singleton-like holder. Call :func:`get_instance` or build your own."""

    _instance: "Optional[AuthStateStore]" = None

    def __init__(self, meta_path: Optional[str] = None) -> None:
        self._lock = RLock()
        self._state: AuthStateLiteral = "disconnected"
        self._phone: Optional[str] = None
        self._phone_code_hash: Optional[str] = None
        self._me_username: Optional[str] = None
        self._last_login_iso: Optional[str] = None
        self._meta_path = Path(_resolve_meta_path(meta_path))
        self._load_meta()

    # ------------------------------------------------------------------
    # Singleton helper
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, meta_path: Optional[str] = None) -> "AuthStateStore":
        if cls._instance is None:
            cls._instance = cls(meta_path=meta_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_meta(self) -> None:
        try:
            if self._meta_path.is_file():
                data = json.loads(self._meta_path.read_text(encoding="utf-8"))
                self._phone = data.get("phone")
                self._me_username = data.get("me_username")
                self._last_login_iso = data.get("last_login_iso")
        except (OSError, json.JSONDecodeError):
            # Corrupt / unreadable meta must not crash the service.
            pass

    def _save_meta(self) -> None:
        try:
            self._meta_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "phone": self._phone,
                "me_username": self._me_username,
                "last_login_iso": self._last_login_iso,
            }
            self._meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state(self) -> AuthState:
        with self._lock:
            age: Optional[int] = None
            if self._last_login_iso:
                try:
                    dt = datetime.fromisoformat(self._last_login_iso)
                    age = int((datetime.now(timezone.utc) - dt).total_seconds())
                except ValueError:
                    age = None
            return AuthState(
                state=self._state,
                phone=self._phone,
                phone_code_hash=self._phone_code_hash,
                session_age_s=age,
                me_username=self._me_username,
            )

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def _transition(self, new_state: AuthStateLiteral) -> None:
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise InvalidTransition(
                f"cannot transition from {self._state!r} to {new_state!r}"
            )
        self._state = new_state

    def set_awaiting_code(self, phone: str, phone_code_hash: str) -> None:
        with self._lock:
            self._transition("awaiting_code")
            self._phone = phone
            self._phone_code_hash = phone_code_hash
            self._save_meta()

    def set_awaiting_2fa(self) -> None:
        with self._lock:
            self._transition("awaiting_2fa")

    def set_authenticated(self, me_username: Optional[str]) -> None:
        with self._lock:
            self._transition("authenticated")
            self._me_username = me_username
            self._phone_code_hash = None  # consumed
            self._last_login_iso = datetime.now(timezone.utc).isoformat()
            self._save_meta()

    def set_disconnected(self) -> None:
        with self._lock:
            self._transition("disconnected")
            self._phone_code_hash = None
            self._save_meta()

    def clear_phone_code_hash(self) -> None:
        with self._lock:
            self._phone_code_hash = None
