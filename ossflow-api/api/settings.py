"""Persistent settings for the BJJ Instructional Processor.

Settings are stored in the unified SQLite DB (``bjj_service_kit.db``).
The DB path defaults to ``/data/db/bjj.db`` but can be overridden via
``BJJ_DB_PATH``. A legacy JSON file at ``CONFIG_DIR/settings.json`` is
auto-imported on first run if present, then renamed to ``settings.json.bak``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from bjj_service_kit.db import init_db, session_scope
from bjj_service_kit.db.models import Setting

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", "/data/config"))
LEGACY_SETTINGS_FILE = CONFIG_DIR / "settings.json"

_DEFAULTS: dict[str, Any] = {
    "library_path": "",
    "voice_profile_default": None,
    "processing_defaults": {
        "chapters": {"dry_run": False, "verbose": True},
        "subtitles": {"verbose": True},
        "translate": {},
        "dubbing": {"use_model_voice": False},
    },
    "custom_prompts": {},
    "telegram_api_id": None,
    "telegram_api_hash": None,
    "openai_api_key": None,
    "translation_provider": "ollama",
    "translation_model": "qwen2.5:7b-instruct-q4_K_M",
    "translation_fallback_provider": "openai",
    # Industry-standard iso-synchronous translation: the translator compacts
    # each ES line to fit the SRT slot so TTS comes out on-time without audio
    # stretch. Works with chat-based providers (Ollama or OpenAI, both budget-aware).
    "translation_dubbing_mode": True,
    "translation_dubbing_cps": 17.0,   # R12: 17 (antes 13). Con tts_engine=elevenlabs la prosodia cloud aguanta densidad Netflix-grade (17 cps es el estándar ES profesional). XTTS requería 13 porque su speed=1.05 fijo no absorbía texto largo sin sonar robótico; ElevenLabs multilingual_v2 pronuncia a cadencia natural y deja el stretcher del pipeline para ajustes finos de slot. Para volver a XTTS, bajar a 13.
    # Motor TTS: "elevenlabs" (cloud, voice cloning, paid) o "piper"
    # (local ONNX, voz preset ES, gratis, sin cloning).
    "tts_engine": "elevenlabs",
    # voice_id pre-registrado en ElevenLabs (PVC o IVC). Ignorado si tts_engine != "elevenlabs".
    "elevenlabs_voice_id": "",
    "elevenlabs_model_id": "eleven_multilingual_v2",
    # Path al modelo Piper ONNX (dentro del contenedor dubbing-generator).
    # Default = es_ES-sharvard-medium baked into the image.
    "piper_model_path": "/models/piper/es_ES-sharvard-medium.onnx",
    # Voz Kokoro-82M (preset ES masculina). Alternativa: em_santa.
    "kokoro_voice": "em_alex",
    # Fish Audio S2-Pro local voice-clone TTS. Engine value "s2pro" enables
    # the dubbing-generator to call the in-container s2.cpp HTTP server.
    # ``s2_voice_profile`` is a basename inside /voices; the dubbing-generator
    # rebuilds the absolute path before calling the server. ``s2_ref_text``
    # MUST exactly match the audio in the voice WAV — drift collapses
    # voice-clone quality.
    "s2_voice_profile": "voice_martin_osborne_24k.wav",
    "s2_ref_text": (
        "nunca te olvidé, nunca, el último beso que me diste todavía está "
        "grabado en mi corazón, por el día todo es más fácil. pero, todavía "
        "sueño contigo."
    ),
    "s2_temperature": 0.8,
    "s2_top_p": 0.8,
    "s2_top_k": 30,
    "s2_max_tokens": 1024,
    # OpenAI post-process for the English SRT produced by WhisperX.
    # Cleans syllable-duplication artifacts and broken mid-clause boundaries
    # while preserving timestamps and block count. Uses openai_api_key.
    "subtitle_postprocess_openai": True,
    # gpt-4o (antes gpt-4o-mini) — el mini no detecta errores de WhisperX
    # tipo "butterflip" por "butterfly". 4o tiene criterio de glosario.
    # Coste marginal: ~1 request por episodio, ~300 tokens in/out.
    "subtitle_postprocess_model": "gpt-4o",
    "author_aliases": {},
}

_TELEGRAM_HASH_RE = re.compile(r"^[0-9a-fA-F]{32}$")

_initialized = False


def _ensure_initialized() -> None:
    """Ensure DB schema exists and import legacy JSON once."""
    global _initialized
    if _initialized:
        return
    try:
        init_db()
    except Exception as exc:
        log.warning("init_db failed: %s", exc)
    _maybe_import_legacy_json()
    _migrate_legacy_translation_settings()
    _initialized = True


def _maybe_import_legacy_json() -> None:
    if not LEGACY_SETTINGS_FILE.exists():
        return
    try:
        data = json.loads(LEGACY_SETTINGS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        with session_scope() as s:
            # Only import keys absent from DB (idempotent)
            existing = {row.key for row in s.query(Setting).all()}
            for k, v in data.items():
                if k in existing:
                    continue
                s.add(Setting(key=k, value=json.dumps(v, ensure_ascii=False)))
        backup = LEGACY_SETTINGS_FILE.with_suffix(".json.bak")
        LEGACY_SETTINGS_FILE.rename(backup)
        log.info("Imported legacy settings.json → DB (backup at %s)", backup)
    except Exception as exc:
        log.warning("Legacy settings import failed: %s", exc)


def _migrate_legacy_translation_settings() -> None:
    """One-shot rewrite de valores legacy (deepl → ollama). Idempotente.

    Solo migra si el usuario tenía explícitamente DeepL configurado. Respeta
    a quien tenga ``translation_provider="openai"`` (no fuerza Ollama).
    """
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
    except Exception as exc:
        log.warning("legacy translation settings migration failed: %s", exc)


def load_settings() -> dict[str, Any]:
    """Return settings merged with defaults.

    Usa ``deepcopy`` porque ``_DEFAULTS`` contiene dicts anidados (p.ej.
    ``processing_defaults.chapters``). Con una copia superficial, mutar el
    dict devuelto contaminaba los defaults compartidos entre requests — un
    bug típico de singleton con estado mutable.
    """
    import copy
    _ensure_initialized()
    merged = copy.deepcopy(_DEFAULTS)
    try:
        with session_scope() as s:
            for row in s.query(Setting).all():
                try:
                    merged[row.key] = json.loads(row.value)
                except json.JSONDecodeError:
                    merged[row.key] = row.value
    except Exception as exc:
        log.warning("load_settings failed, returning defaults: %s", exc)
    return merged


def save_settings(data: dict[str, Any]) -> None:
    """Persist settings to DB (upsert per key)."""
    _ensure_initialized()
    with session_scope() as s:
        for k, v in data.items():
            payload = json.dumps(v, ensure_ascii=False)
            row = s.get(Setting, k)
            if row is None:
                s.add(Setting(key=k, value=payload))
            else:
                row.value = payload
    log.info("Settings saved to DB")


def get_library_path() -> Optional[str]:
    settings = load_settings()
    lp = settings.get("library_path", "")
    return lp if lp else None


def get_setting(key: str) -> Any:
    """Read a single settings value (merged with defaults)."""
    return load_settings().get(key)


# ---------------------------------------------------------------------------
# Routes (API contract unchanged)
# ---------------------------------------------------------------------------

_SECRET_KEYS = {"openai_api_key", "telegram_api_hash", "elevenlabs_api_key", "deepl_api_key"}


def _mask_secrets(data: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in data.items():
        if k in _SECRET_KEYS or k.endswith("_api_key"):
            out[k] = "***" if v else None
        else:
            out[k] = v
    return out


@router.get("")
async def get_settings():
    return _mask_secrets(load_settings())


@router.get("/internal")
async def get_settings_internal(request: Request):
    """Unmasked settings for in-stack service-to-service calls.

    Telegram-fetcher needs the real ``telegram_api_hash`` to call the
    Telethon API; it cannot use the public masked endpoint (would send
    "***" to Telegram → ApiIdInvalidError). Restricted to private Docker
    network ranges so it can never be reached from the host browser.
    """
    client_host = request.client.host if request.client else ""
    # Docker bridge networks live in 172.16.0.0/12 (default) or 10.x for
    # custom networks. Loopback is allowed for local dev / curl from
    # processor-api's own container.
    allowed = (
        client_host.startswith("172.")
        or client_host.startswith("10.")
        or client_host.startswith("192.168.")
        or client_host in ("127.0.0.1", "localhost", "::1", "testclient")
    )
    if not allowed:
        return JSONResponse(
            {"error": "internal endpoint, network-restricted"},
            status_code=403,
        )
    return load_settings()


@router.put("")
async def put_settings(request: Request):
    body = await request.json()

    if not isinstance(body, dict):
        return JSONResponse({"error": "Request body must be a JSON object"}, status_code=422)

    current = load_settings()

    if "library_path" in body:
        lp = body["library_path"]
        if not isinstance(lp, str):
            return JSONResponse({"error": "library_path must be a string"}, status_code=422)
        current["library_path"] = lp

    if "voice_profile_default" in body:
        current["voice_profile_default"] = body["voice_profile_default"]

    if "translation_dubbing_cps" in body:
        # Chars-per-second target para la traducción iso-síncrona.
        # Rango razonable: 12-18. Más bajo = texto ES más compacto
        # (el LLM reescribe manteniendo significado) → cabe mejor en
        # el slot EN. Más alto = traducción más literal pero puede
        # desbordar al dub.
        tc = body["translation_dubbing_cps"]
        if not isinstance(tc, (int, float)) or not (8 <= tc <= 25):
            return JSONResponse(
                {"error": "translation_dubbing_cps must be number in [8, 25]"},
                status_code=422,
            )
        current["translation_dubbing_cps"] = float(tc)

    if "processing_defaults" in body:
        pd = body["processing_defaults"]
        if not isinstance(pd, dict):
            return JSONResponse({"error": "processing_defaults must be a JSON object"}, status_code=422)
        current["processing_defaults"] = pd

    if "telegram_api_id" in body:
        tid = body["telegram_api_id"]
        if tid is not None and not (isinstance(tid, int) and not isinstance(tid, bool)):
            return JSONResponse({"error": "telegram_api_id must be an integer or null"}, status_code=422)
        current["telegram_api_id"] = tid

    if "telegram_api_hash" in body:
        th = body["telegram_api_hash"]
        if th is not None:
            if not isinstance(th, str) or not _TELEGRAM_HASH_RE.match(th):
                return JSONResponse({"error": "telegram_api_hash must be a 32-char hex string or null"}, status_code=422)
        current["telegram_api_hash"] = th

    if "openai_api_key" in body:
        ok = body["openai_api_key"]
        if ok is not None and not isinstance(ok, str):
            return JSONResponse({"error": "openai_api_key must be a string or null"}, status_code=422)
        if ok != "***":  # sentinel from masked GET response — ignore
            current["openai_api_key"] = ok.strip() if isinstance(ok, str) else ok

    if "translation_provider" in body:
        val = body["translation_provider"]
        if val not in ("ollama", "openai"):
            return JSONResponse(
                {"error": "translation_provider must be 'ollama' or 'openai'"},
                status_code=422,
            )
        current["translation_provider"] = val

    if "translation_model" in body:
        v = body["translation_model"]
        if v is not None and not isinstance(v, str):
            return JSONResponse({"error": "translation_model must be a string or null"}, status_code=422)
        current["translation_model"] = v.strip() if isinstance(v, str) else v

    if "translation_fallback_provider" in body:
        val = body["translation_fallback_provider"]
        if val not in ("", "ollama", "openai", None):
            return JSONResponse(
                {"error": "translation_fallback_provider must be '', 'ollama', 'openai' or null"},
                status_code=422,
            )
        current["translation_fallback_provider"] = val if val else None

    if "author_aliases" in body:
        aa = body["author_aliases"]
        if not isinstance(aa, dict):
            return JSONResponse({"error": "author_aliases must be a JSON object"}, status_code=422)
        cleaned: dict[str, str] = {}
        for k, v in aa.items():
            if not isinstance(k, str) or not isinstance(v, str):
                return JSONResponse({"error": "author_aliases keys and values must be strings"}, status_code=422)
            k2, v2 = k.strip(), v.strip()
            if k2 and v2:
                cleaned[k2] = v2
        current["author_aliases"] = cleaned

    if "subtitle_postprocess_openai" in body:
        v = body["subtitle_postprocess_openai"]
        if not isinstance(v, bool):
            return JSONResponse({"error": "subtitle_postprocess_openai must be a boolean"}, status_code=422)
        current["subtitle_postprocess_openai"] = v

    if "subtitle_postprocess_model" in body:
        v = body["subtitle_postprocess_model"]
        if v is not None and not isinstance(v, str):
            return JSONResponse({"error": "subtitle_postprocess_model must be a string or null"}, status_code=422)
        current["subtitle_postprocess_model"] = v.strip() if isinstance(v, str) else v

    if "custom_prompts" in body:
        cp = body["custom_prompts"]
        if not isinstance(cp, dict):
            return JSONResponse({"error": "custom_prompts must be a JSON object"}, status_code=422)
        current["custom_prompts"] = cp

    if "tts_engine" in body:
        te = body["tts_engine"]
        if not isinstance(te, str) or te.strip().lower() not in (
            "s2pro", "elevenlabs", "piper", "kokoro",
        ):
            return JSONResponse(
                {"error": "tts_engine must be 's2pro', 'elevenlabs', 'piper' or 'kokoro'"},
                status_code=422,
            )
        current["tts_engine"] = te.strip().lower()

    if "elevenlabs_voice_id" in body:
        v = body["elevenlabs_voice_id"]
        if v is not None and not isinstance(v, str):
            return JSONResponse({"error": "elevenlabs_voice_id must be a string or null"}, status_code=422)
        current["elevenlabs_voice_id"] = v.strip() if isinstance(v, str) else v

    if "elevenlabs_model_id" in body:
        v = body["elevenlabs_model_id"]
        if v is not None and not isinstance(v, str):
            return JSONResponse({"error": "elevenlabs_model_id must be a string or null"}, status_code=422)
        current["elevenlabs_model_id"] = v.strip() if isinstance(v, str) else v

    if "piper_model_path" in body:
        v = body["piper_model_path"]
        if v is not None and not isinstance(v, str):
            return JSONResponse({"error": "piper_model_path must be a string or null"}, status_code=422)
        current["piper_model_path"] = v.strip() if isinstance(v, str) else v

    if "kokoro_voice" in body:
        v = body["kokoro_voice"]
        if v is not None and not isinstance(v, str):
            return JSONResponse({"error": "kokoro_voice must be a string or null"}, status_code=422)
        if isinstance(v, str) and v.strip() and v.strip() not in ("em_alex", "em_santa"):
            return JSONResponse(
                {"error": "kokoro_voice must be 'em_alex' or 'em_santa'"},
                status_code=422,
            )
        current["kokoro_voice"] = v.strip() if isinstance(v, str) else v

    if "s2_voice_profile" in body:
        v = body["s2_voice_profile"]
        if v is not None and not isinstance(v, str):
            return JSONResponse(
                {"error": "s2_voice_profile must be a string or null"},
                status_code=422,
            )
        current["s2_voice_profile"] = v.strip() if isinstance(v, str) else v

    if "s2_ref_text" in body:
        v = body["s2_ref_text"]
        if not isinstance(v, str) or not v.strip():
            return JSONResponse(
                {"error": "s2_ref_text must be a non-empty string"},
                status_code=422,
            )
        current["s2_ref_text"] = v

    for fkey, lo, hi in (
        ("s2_temperature", 0.1, 1.5),
        ("s2_top_p", 0.1, 1.0),
    ):
        if fkey in body:
            v = body[fkey]
            if not isinstance(v, (int, float)) or not lo <= float(v) <= hi:
                return JSONResponse(
                    {"error": f"{fkey} must be a number in [{lo}, {hi}]"},
                    status_code=422,
                )
            current[fkey] = float(v)

    if "s2_top_k" in body:
        v = body["s2_top_k"]
        if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 200:
            return JSONResponse(
                {"error": "s2_top_k must be an integer in [1, 200]"},
                status_code=422,
            )
        current["s2_top_k"] = v

    if "s2_max_tokens" in body:
        v = body["s2_max_tokens"]
        if not isinstance(v, int) or isinstance(v, bool) or not 128 <= v <= 2048:
            return JSONResponse(
                {"error": "s2_max_tokens must be an integer in [128, 2048]"},
                status_code=422,
            )
        current["s2_max_tokens"] = v

    save_settings(current)
    return current
