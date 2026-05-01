"""Servicio de settings: carga/persiste settings en SQLite + masking + validación.

Mantiene el comportamiento histórico del antiguo ``api/settings.py``: idempotencia
en init, deepcopy de defaults, masking de secrets, e import one-shot del legacy
``settings.json`` con backup. La validación del PUT también es idéntica para
no cambiar el contrato HTTP.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Optional

from ossflow_service_kit.db import init_db, session_scope
from ossflow_service_kit.db.models import Setting

from .schemas import (
    DEFAULTS,
    LEGACY_SETTINGS_FILE,
    SECRET_KEYS,
    TELEGRAM_HASH_RE,
)

log = logging.getLogger(__name__)


class SettingsService:
    """Encapsula init, lectura, escritura y validación de settings.

    Mantiene un único flag ``_initialized`` por instancia para hacer la
    inicialización idempotente. ``dependencies.py`` expone un singleton de
    proceso para que la BD solo se inicialice una vez.
    """

    def __init__(self) -> None:
        self._initialized = False

    # --- Inicialización ----------------------------------------------------

    def ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            init_db()
        except Exception as exc:  # noqa: BLE001
            log.warning("init_db failed: %s", exc)
        self._maybe_import_legacy_json()
        self._migrate_legacy_translation_settings()
        self._migrate_legacy_tts_settings()
        self._initialized = True

    @staticmethod
    def _maybe_import_legacy_json() -> None:
        if not LEGACY_SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(LEGACY_SETTINGS_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            with session_scope() as s:
                existing = {row.key for row in s.query(Setting).all()}
                for k, v in data.items():
                    if k in existing:
                        continue
                    s.add(Setting(key=k, value=json.dumps(v, ensure_ascii=False)))
            backup = LEGACY_SETTINGS_FILE.with_suffix(".json.bak")
            LEGACY_SETTINGS_FILE.rename(backup)
            log.info("Imported legacy settings.json → DB (backup at %s)", backup)
        except Exception as exc:  # noqa: BLE001
            log.warning("Legacy settings import failed: %s", exc)

    @staticmethod
    def _migrate_legacy_translation_settings() -> None:
        """One-shot rewrite de valores legacy (deepl → ollama). Idempotente."""
        try:
            with session_scope() as s:
                row = s.get(Setting, "translation_provider")
                if row is not None and json.loads(row.value) == "deepl":
                    row.value = json.dumps("ollama")
                    model_row = s.get(Setting, "translation_model")
                    if model_row is not None:
                        model_row.value = json.dumps("qwen2.5:7b-instruct-q4_K_M")

                fb_row = s.get(Setting, "translation_fallback_provider")
                if fb_row is not None and json.loads(fb_row.value) == "deepl":
                    fb_row.value = json.dumps("openai")

                deepl_row = s.get(Setting, "deepl_api_key")
                if deepl_row is not None:
                    s.delete(deepl_row)
        except Exception as exc:  # noqa: BLE001
            log.warning("legacy translation settings migration failed: %s", exc)

    # Settings TTS obsoletos tras la limpieza T22.5 (eliminación de los
    # motores ElevenLabs/Piper/Kokoro). Las filas se borran al primer
    # arranque tras el deploy. Idempotente: ejecuciones posteriores son
    # no-op porque las filas ya no están.
    _LEGACY_TTS_KEYS = (
        "tts_engine",
        "elevenlabs_voice_id",
        "elevenlabs_model_id",
        "elevenlabs_api_key",
        "piper_model_path",
        "kokoro_voice",
    )

    @classmethod
    def _migrate_legacy_tts_settings(cls) -> None:
        """One-shot delete de settings de motores TTS eliminados (T22.5).

        Mismo patrón que ``_migrate_legacy_translation_settings``: borra las
        filas obsoletas para que el sistema no tenga estado huérfano. Si
        un usuario tenía ``tts_engine="elevenlabs"`` persistido, esa fila
        desaparece y al próximo ``load()`` el merge con ``DEFAULTS`` no la
        reintroduce (no existe en ``DEFAULTS``).
        """
        try:
            with session_scope() as s:
                for key in cls._LEGACY_TTS_KEYS:
                    row = s.get(Setting, key)
                    if row is not None:
                        s.delete(row)
        except Exception as exc:  # noqa: BLE001
            log.warning("legacy TTS settings migration failed: %s", exc)

    # --- Lectura -----------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Devuelve settings mergeados con defaults (deepcopy para evitar
        contaminar los defaults compartidos)."""
        self.ensure_initialized()
        merged = copy.deepcopy(DEFAULTS)
        try:
            with session_scope() as s:
                for row in s.query(Setting).all():
                    try:
                        merged[row.key] = json.loads(row.value)
                    except json.JSONDecodeError:
                        merged[row.key] = row.value
        except Exception as exc:  # noqa: BLE001
            log.warning("load_settings failed, returning defaults: %s", exc)
        return merged

    def save(self, data: dict[str, Any]) -> None:
        self.ensure_initialized()
        with session_scope() as s:
            for k, v in data.items():
                payload = json.dumps(v, ensure_ascii=False)
                row = s.get(Setting, k)
                if row is None:
                    s.add(Setting(key=k, value=payload))
                else:
                    row.value = payload
        log.info("Settings saved to DB")

    def get_library_path(self) -> Optional[str]:
        lp = self.load().get("library_path", "")
        return lp if lp else None

    def get(self, key: str) -> Any:
        return self.load().get(key)

    # --- Masking -----------------------------------------------------------

    @staticmethod
    def mask_secrets(data: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in data.items():
            if k in SECRET_KEYS or k.endswith("_api_key"):
                out[k] = "***" if v else None
            else:
                out[k] = v
        return out

    # --- Validación + actualización ---------------------------------------

    def update(self, body: Any) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Valida y persiste un PUT.

        Devuelve ``(error_msg, settings_actualizados)``: si hay error de
        validación, ``settings_actualizados`` es ``None`` y el caller (router)
        responde 422 con el mensaje. Si todo OK, ``error_msg`` es ``None`` y
        devuelve los settings finales.
        """
        if not isinstance(body, dict):
            return "Request body must be a JSON object", None

        current = self.load()

        if "library_path" in body:
            lp = body["library_path"]
            if not isinstance(lp, str):
                return "library_path must be a string", None
            current["library_path"] = lp

        if "voice_profile_default" in body:
            current["voice_profile_default"] = body["voice_profile_default"]

        if "translation_dubbing_cps" in body:
            tc = body["translation_dubbing_cps"]
            if not isinstance(tc, (int, float)) or not (8 <= tc <= 25):
                return "translation_dubbing_cps must be number in [8, 25]", None
            current["translation_dubbing_cps"] = float(tc)

        if "processing_defaults" in body:
            pd = body["processing_defaults"]
            if not isinstance(pd, dict):
                return "processing_defaults must be a JSON object", None
            current["processing_defaults"] = pd

        if "telegram_api_id" in body:
            tid = body["telegram_api_id"]
            if tid is not None and not (isinstance(tid, int) and not isinstance(tid, bool)):
                return "telegram_api_id must be an integer or null", None
            current["telegram_api_id"] = tid

        if "telegram_api_hash" in body:
            th = body["telegram_api_hash"]
            if th is not None:
                if not isinstance(th, str) or not TELEGRAM_HASH_RE.match(th):
                    return "telegram_api_hash must be a 32-char hex string or null", None
            current["telegram_api_hash"] = th

        if "openai_api_key" in body:
            ok = body["openai_api_key"]
            if ok is not None and not isinstance(ok, str):
                return "openai_api_key must be a string or null", None
            if ok != "***":  # sentinel from masked GET response — ignore
                current["openai_api_key"] = ok.strip() if isinstance(ok, str) else ok

        if "translation_provider" in body:
            val = body["translation_provider"]
            if val not in ("ollama", "openai"):
                return "translation_provider must be 'ollama' or 'openai'", None
            current["translation_provider"] = val

        if "translation_model" in body:
            v = body["translation_model"]
            if v is not None and not isinstance(v, str):
                return "translation_model must be a string or null", None
            current["translation_model"] = v.strip() if isinstance(v, str) else v

        if "translation_fallback_provider" in body:
            val = body["translation_fallback_provider"]
            if val not in ("", "ollama", "openai", None):
                return "translation_fallback_provider must be '', 'ollama', 'openai' or null", None
            current["translation_fallback_provider"] = val if val else None

        if "author_aliases" in body:
            aa = body["author_aliases"]
            if not isinstance(aa, dict):
                return "author_aliases must be a JSON object", None
            cleaned: dict[str, str] = {}
            for k, v in aa.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    return "author_aliases keys and values must be strings", None
                k2, v2 = k.strip(), v.strip()
                if k2 and v2:
                    cleaned[k2] = v2
            current["author_aliases"] = cleaned

        if "subtitle_postprocess_openai" in body:
            v = body["subtitle_postprocess_openai"]
            if not isinstance(v, bool):
                return "subtitle_postprocess_openai must be a boolean", None
            current["subtitle_postprocess_openai"] = v

        if "subtitle_postprocess_model" in body:
            v = body["subtitle_postprocess_model"]
            if v is not None and not isinstance(v, str):
                return "subtitle_postprocess_model must be a string or null", None
            current["subtitle_postprocess_model"] = v.strip() if isinstance(v, str) else v

        if "custom_prompts" in body:
            cp = body["custom_prompts"]
            if not isinstance(cp, dict):
                return "custom_prompts must be a JSON object", None
            current["custom_prompts"] = cp

        if "s2_voice_profile" in body:
            v = body["s2_voice_profile"]
            if v is not None and not isinstance(v, str):
                return "s2_voice_profile must be a string or null", None
            current["s2_voice_profile"] = v.strip() if isinstance(v, str) else v

        if "s2_ref_text" in body:
            v = body["s2_ref_text"]
            if not isinstance(v, str) or not v.strip():
                return "s2_ref_text must be a non-empty string", None
            current["s2_ref_text"] = v

        for fkey, lo, hi in (
            ("s2_temperature", 0.1, 1.5),
            ("s2_top_p", 0.1, 1.0),
        ):
            if fkey in body:
                v = body[fkey]
                if not isinstance(v, (int, float)) or not lo <= float(v) <= hi:
                    return f"{fkey} must be a number in [{lo}, {hi}]", None
                current[fkey] = float(v)

        if "s2_top_k" in body:
            v = body["s2_top_k"]
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 200:
                return "s2_top_k must be an integer in [1, 200]", None
            current["s2_top_k"] = v

        if "s2_max_tokens" in body:
            v = body["s2_max_tokens"]
            if not isinstance(v, int) or isinstance(v, bool) or not 128 <= v <= 2048:
                return "s2_max_tokens must be an integer in [128, 2048]", None
            current["s2_max_tokens"] = v

        if "s2_quantization" in body:
            v = body["s2_quantization"]
            if not isinstance(v, str) or v.strip().lower() not in ("q4_k_m", "q6_k"):
                return "s2_quantization must be 'q4_k_m' or 'q6_k'", None
            current["s2_quantization"] = v.strip().lower()

        self.save(current)
        return None, current
