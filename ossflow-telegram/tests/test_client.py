"""Tests for TelegramService — all Telethon calls are mocked."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from telegram_fetcher.auth_store import AuthStateStore
from telegram_fetcher.client import FLOOD_THRESHOLD_S, TelegramService
from telegram_fetcher.errors import (
    AuthFailedError,
    AuthRequiredError,
    RateLimitError,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fake Telethon exception classes. The translator inspects type(exc).__name__.
# ---------------------------------------------------------------------------


class FloodWaitError(Exception):
    def __init__(self, seconds: int) -> None:
        super().__init__(f"flood {seconds}")
        self.seconds = seconds


class SessionPasswordNeededError(Exception):
    pass


class PhoneCodeInvalidError(Exception):
    pass


class AuthKeyUnregisteredError(Exception):
    pass


class FakeSentCode:
    def __init__(self, h: str = "HASH") -> None:
        self.phone_code_hash = h


class FakeMe:
    def __init__(self, username: Optional[str] = "alice") -> None:
        self.username = username


class FakeClient:
    def __init__(self, *, authorized_after_signin: bool = True) -> None:
        self.connected = False
        self._authorized = False
        self._authorized_after_signin = authorized_after_signin
        self.last_code: Any = None
        # Programmable overrides
        self.sign_in_exc: Optional[BaseException] = None
        self.send_code_exc: Optional[BaseException] = None

    def is_connected(self) -> bool:
        return self.connected

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def is_user_authorized(self) -> bool:
        return self._authorized

    async def send_code_request(self, phone: str) -> FakeSentCode:
        if self.send_code_exc is not None:
            raise self.send_code_exc
        return FakeSentCode("HASH_ABC")

    async def sign_in(self, phone: str = None, code: str = None, *,
                      password: str = None, phone_code_hash: str = None) -> Any:
        if self.sign_in_exc is not None:
            exc = self.sign_in_exc
            self.sign_in_exc = None  # consume
            raise exc
        self.last_code = code
        self._authorized = self._authorized_after_signin
        return FakeMe()

    async def get_me(self) -> FakeMe:
        return FakeMe("alice")

    async def log_out(self) -> bool:
        self._authorized = False
        return True


@pytest.fixture
def store(tmp_path: Path) -> AuthStateStore:
    AuthStateStore.reset_instance()
    return AuthStateStore(meta_path=str(tmp_path / "m.json"))


def _svc(store: AuthStateStore, fake: FakeClient, session_dir: Path) -> TelegramService:
    svc = TelegramService(
        api_id=1, api_hash="hash",
        session_path=str(session_dir / "sess"),
        auth_store=store,
    )
    svc.set_client_factory(lambda sp, aid, ah: fake)
    return svc


class TestAuthFlow:
    async def test_send_code_transitions_to_awaiting_code(
        self, store: AuthStateStore, tmp_path: Path
    ) -> None:
        fake = FakeClient()
        svc = _svc(store, fake, tmp_path)
        h = await svc.send_code("+34600000000")
        assert h == "HASH_ABC"
        assert store.get_state().state == "awaiting_code"
        assert store.get_state().phone_code_hash == "HASH_ABC"

    async def test_sign_in_code_authenticates(
        self, store: AuthStateStore, tmp_path: Path
    ) -> None:
        fake = FakeClient()
        svc = _svc(store, fake, tmp_path)
        await svc.send_code("+1")
        await svc.sign_in_code("+1", "12345", "HASH_ABC")
        assert store.get_state().state == "authenticated"
        assert store.get_state().me_username == "alice"

    async def test_sign_in_requires_2fa(self, store: AuthStateStore, tmp_path: Path) -> None:
        fake = FakeClient()
        svc = _svc(store, fake, tmp_path)
        await svc.send_code("+1")
        fake.sign_in_exc = SessionPasswordNeededError()
        await svc.sign_in_code("+1", "12345", "HASH_ABC")
        assert store.get_state().state == "awaiting_2fa"
        await svc.sign_in_2fa("pw")
        assert store.get_state().state == "authenticated"

    async def test_invalid_code_maps_to_auth_failed(
        self, store: AuthStateStore, tmp_path: Path
    ) -> None:
        fake = FakeClient()
        svc = _svc(store, fake, tmp_path)
        await svc.send_code("+1")
        fake.sign_in_exc = PhoneCodeInvalidError()
        with pytest.raises(AuthFailedError):
            await svc.sign_in_code("+1", "0", "HASH_ABC")

    async def test_missing_credentials_raises_auth_required(
        self, store: AuthStateStore, tmp_path: Path
    ) -> None:
        svc = TelegramService(api_id=None, api_hash=None, auth_store=store,
                              session_path=str(tmp_path / "s"))
        with pytest.raises(AuthRequiredError):
            await svc.connect()


class TestErrorTranslation:
    async def test_floodwait_long_becomes_rate_limit(
        self, store: AuthStateStore, tmp_path: Path
    ) -> None:
        fake = FakeClient()
        fake.send_code_exc = FloodWaitError(FLOOD_THRESHOLD_S + 30)
        svc = _svc(store, fake, tmp_path)
        with pytest.raises(RateLimitError) as ei:
            await svc.send_code("+1")
        assert ei.value.retry_after_s == FLOOD_THRESHOLD_S + 30

    async def test_auth_key_unregistered_becomes_auth_required(
        self, store: AuthStateStore, tmp_path: Path
    ) -> None:
        fake = FakeClient()
        svc = _svc(store, fake, tmp_path)
        await svc.send_code("+1")
        fake.sign_in_exc = AuthKeyUnregisteredError()
        with pytest.raises(AuthRequiredError):
            await svc.sign_in_code("+1", "0", "HASH_ABC")
        assert store.get_state().state == "disconnected"


class TestLogout:
    async def test_logout_resets_state(self, store: AuthStateStore, tmp_path: Path) -> None:
        fake = FakeClient()
        svc = _svc(store, fake, tmp_path)
        await svc.send_code("+1")
        await svc.sign_in_code("+1", "12345", "HASH_ABC")
        assert store.get_state().state == "authenticated"
        await svc.logout()
        assert store.get_state().state == "disconnected"
