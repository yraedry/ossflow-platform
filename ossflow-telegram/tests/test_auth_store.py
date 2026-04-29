"""Tests for AuthStateStore — transitions + JSON meta persistence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from telegram_fetcher.auth_store import AuthStateStore, InvalidTransition


@pytest.fixture
def store(tmp_path: Path) -> AuthStateStore:
    AuthStateStore.reset_instance()
    return AuthStateStore(meta_path=str(tmp_path / "auth_meta.json"))


class TestTransitions:
    def test_initial_state_is_disconnected(self, store: AuthStateStore) -> None:
        assert store.get_state().state == "disconnected"

    def test_full_happy_path(self, store: AuthStateStore) -> None:
        store.set_awaiting_code("+34600000000", "HASH_ABC")
        assert store.get_state().state == "awaiting_code"
        assert store.get_state().phone_code_hash == "HASH_ABC"
        store.set_awaiting_2fa()
        assert store.get_state().state == "awaiting_2fa"
        store.set_authenticated(me_username="alice")
        state = store.get_state()
        assert state.state == "authenticated"
        assert state.me_username == "alice"
        # phone_code_hash is consumed after auth.
        assert state.phone_code_hash is None

    def test_direct_code_to_authenticated(self, store: AuthStateStore) -> None:
        store.set_awaiting_code("+1", "H")
        store.set_authenticated("bob")
        assert store.get_state().state == "authenticated"

    def test_invalid_transition_rejected(self, store: AuthStateStore) -> None:
        with pytest.raises(InvalidTransition):
            store.set_awaiting_2fa()  # disconnected -> awaiting_2fa is invalid

    def test_logout_from_authenticated(self, store: AuthStateStore) -> None:
        store.set_awaiting_code("+1", "H")
        store.set_authenticated("bob")
        store.set_disconnected()
        assert store.get_state().state == "disconnected"
        assert store.get_state().phone_code_hash is None


class TestPersistence:
    def test_meta_survives_reinstantiation(self, tmp_path: Path) -> None:
        meta = tmp_path / "m.json"
        AuthStateStore.reset_instance()
        s1 = AuthStateStore(meta_path=str(meta))
        s1.set_awaiting_code("+34999", "H")
        s1.set_authenticated("alice")
        assert meta.is_file()
        data = json.loads(meta.read_text(encoding="utf-8"))
        assert data["phone"] == "+34999"
        assert data["me_username"] == "alice"
        assert data["last_login_iso"]

        AuthStateStore.reset_instance()
        s2 = AuthStateStore(meta_path=str(meta))
        # phone/username restored from disk; state resets to disconnected though.
        st = s2.get_state()
        assert st.phone == "+34999"
        assert st.me_username == "alice"
        assert st.state == "disconnected"  # in-memory only

    def test_phone_code_hash_never_persisted(self, tmp_path: Path) -> None:
        meta = tmp_path / "m.json"
        AuthStateStore.reset_instance()
        s = AuthStateStore(meta_path=str(meta))
        s.set_awaiting_code("+1", "SECRET_HASH")
        data = json.loads(meta.read_text(encoding="utf-8"))
        assert "SECRET_HASH" not in json.dumps(data)
        assert "phone_code_hash" not in data
