def test_s2pro_defaults():
    from dubbing_generator.config import DubbingConfig

    cfg = DubbingConfig()
    assert cfg.tts_engine == "s2pro"
    assert cfg.s2_server_host == "127.0.0.1"
    assert cfg.s2_server_port == 3030
    assert cfg.s2_gguf_path == "/models/s2pro/s2-pro-q6_k.gguf"
    assert cfg.s2_tokenizer_path == "/models/s2pro/tokenizer.json"
    assert cfg.s2_ref_audio_path == "/voices/voice_martin_osborne_24k.wav"
    assert "nunca te olvidé" in cfg.s2_ref_text
    assert cfg.s2_temperature == 0.8
    assert cfg.s2_top_p == 0.8
    assert cfg.s2_top_k == 30
    assert cfg.s2_max_tokens == 1024
    assert cfg.s2_request_timeout == 180.0
    assert cfg.s2_vulkan_device == 0
    assert cfg.s2_health_timeout_s == 60.0


def test_s2pro_engine_in_allowed_list():
    from dubbing_generator.config import DubbingConfig

    cfg = DubbingConfig(tts_engine="s2pro")
    assert cfg.tts_engine == "s2pro"
