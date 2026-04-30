"""Defaults y constantes del módulo settings."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

CONFIG_DIR: Path = Path(os.environ.get("CONFIG_DIR", "/data/config"))
LEGACY_SETTINGS_FILE: Path = CONFIG_DIR / "settings.json"

TELEGRAM_HASH_RE = re.compile(r"^[0-9a-fA-F]{32}$")

SECRET_KEYS: set[str] = {
    "openai_api_key",
    "telegram_api_hash",
    "elevenlabs_api_key",
    "deepl_api_key",
}

DEFAULTS: dict[str, Any] = {
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
