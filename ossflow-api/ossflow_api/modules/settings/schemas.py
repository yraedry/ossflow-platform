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
    # Chars-per-second target para la traducción iso-síncrona del SRT.
    # 17 cps es el estándar ES profesional Netflix-grade que aguanta la
    # prosodia neutral del clon S2-Pro sin necesidad de stretching de
    # audio en el muxing.
    "translation_dubbing_cps": 17.0,
    # Fish Audio S2-Pro local voice-clone TTS — único motor TTS soportado
    # tras T22.5 (eliminados ElevenLabs/Piper/Kokoro).
    # ``s2_voice_profile`` es un basename dentro de /voices; el
    # dubbing-generator reconstruye el path absoluto antes de llamar al
    # servidor s2.cpp embebido. ``s2_ref_text`` DEBE coincidir
    # EXACTAMENTE con el audio del WAV de referencia — la deriva colapsa
    # la calidad del clon de voz.
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
    # Cuantización del modelo GGUF S2-Pro:
    #   q4_k_m: ~3 GB VRAM, calidad menor, encaja en GPUs de 6 GB.
    #   q6_k:   ~5 GB VRAM, mejor calidad, recomendada en ≥8 GB.
    # El path absoluto se construye en pipeline.py como
    # /models/s2pro/s2-pro-{quant}.gguf antes de pasarlo al dubbing.
    "s2_quantization": "q6_k",
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
