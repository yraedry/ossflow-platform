"""Tests for DubbingConfig defaults and constraints."""

from dubbing_generator.config import DubbingConfig


def test_tts_defaults():
    cfg = DubbingConfig()
    # Ronda 9 — Iter 3: vuelve a 1.05 (baseline). Iter 2 bajó a 1.02 sin
    # reducir saltos de timbre — son intrínsecos al XTTS.
    assert cfg.tts_speed == 1.05
    # R10: revertidos a cerca de defaults XTTS tras detectar balbuceo.
    assert cfg.tts_temperature == 0.75
    assert cfg.tts_repetition_penalty == 5.0
    assert cfg.tts_top_p == 0.85
    assert cfg.target_language == "es"


def test_compression_bounds():
    cfg = DubbingConfig()
    # 1.25 (antes 1.30): bajamos techo base; el tail nudge sube a 1.35
    # solo en el tramo final con deriva — mayoría de frases no notan
    # cambio y evitamos artefactos F0 del stretcher.
    assert cfg.max_compression_ratio == 1.25
    assert cfg.min_compression_ratio == 0.90


def test_voice_sample_duration():
    cfg = DubbingConfig()
    assert cfg.voice_sample_duration == 12.0


def test_ducking_defaults():
    cfg = DubbingConfig()
    assert cfg.ducking_bg_volume == 0.12
    assert cfg.ducking_fg_volume == 1.6
    assert cfg.ducking_fade_ms == 180


def test_drift_defaults():
    cfg = DubbingConfig()
    assert cfg.drift_check_interval == 8
    assert cfg.drift_threshold_ms == 250.0
    assert cfg.speed_min == 0.95
    assert cfg.speed_max == 1.10
    assert cfg.constant_speed is True


def test_use_cloned_voice_by_default():
    cfg = DubbingConfig()
    assert cfg.use_model_voice is False


def test_custom_overrides():
    cfg = DubbingConfig(tts_speed=0.95, max_compression_ratio=1.3)
    assert cfg.tts_speed == 0.95
    assert cfg.max_compression_ratio == 1.3


def test_xtts_defaults():
    cfg = DubbingConfig()
    assert cfg.tts_engine == "xttsv2"
    assert cfg.xtts_model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
    assert cfg.xtts_config_path == ""
    assert cfg.xtts_checkpoint_dir == ""
    assert cfg.xtts_use_deepspeed is False
    # Ronda 8: code-switching DESACTIVADO por defecto. XTTS-v2 no
    # soporta alternar ``language`` entre inferencias dentro del mismo
    # discurso sin alucinar a chino/japonés (el GPT-autoregresivo
    # arrastra estado). Fix definitivo: single-span ES para todo.
    assert cfg.xtts_code_switching is False
    assert cfg.xtts_en_terms_extra == ()


def test_sync_defaults():
    cfg = DubbingConfig()
    assert cfg.inter_phrase_pad_ms == 20
    assert cfg.max_overflow_ms == 250
    # Tail extension habilitada para no perder info al final del vídeo.
    assert cfg.allow_video_tail_extension is True
    # R11: 6000 ms — tras R10.2 cortaba el último bloque SRT. Usuario
    # prioriza preservar contenido > deriva visual corta.
    assert cfg.video_tail_extension_max_ms == 6000


def test_tail_speed_nudge_defaults():
    cfg = DubbingConfig()
    # R11: nudge más agresivo y temprano para evitar que el
    # tail_extension=6000 se agote y corte contenido.
    assert cfg.tail_speed_nudge_window_ms == 45000
    assert cfg.tail_speed_nudge_trigger_ms == 400
    assert cfg.tail_speed_nudge_max_ratio == 1.45


def test_elevenlabs_defaults():
    cfg = DubbingConfig()
    assert cfg.elevenlabs_voice_id == "LlZr3QuzbW4WrPjgATHG"
    assert cfg.elevenlabs_model_id == "eleven_multilingual_v2"
    assert cfg.elevenlabs_stability == 0.5
    assert cfg.elevenlabs_similarity_boost == 0.75
    assert cfg.elevenlabs_style == 0.0
    assert cfg.elevenlabs_use_speaker_boost is True
    assert cfg.elevenlabs_output_format == "pcm_24000"
    assert cfg.elevenlabs_api_key_env == "ELEVENLABS_API_KEY"
    assert cfg.elevenlabs_request_timeout == 60.0
