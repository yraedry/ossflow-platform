"""Centralized configuration for the dubbing generator."""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    """Read env var, treating empty string as 'use default' (compose passes
    through `${VAR:-}` so the key is always present, often empty)."""
    val = os.environ.get(name) or ""
    return val if val else default


@dataclass
class DubbingConfig:
    """All dubbing pipeline parameters in one place."""

    # ------------------------------------------------------------------
    # TTS — default engine: S2-Pro (fish-speech via s2.cpp HTTP server).
    # ------------------------------------------------------------------
    # Voice cloning over a reference WAV + matching transcript pair. The
    # XTTS / ElevenLabs / Piper / Kokoro fields below are kept as live
    # defaults so DubbingConfig stays backwards-compatible with the
    # callers that still reference them, but only the s2_* fields and
    # generic merge/crossfade fields are read by the active code path.
    # See ``tts/synthesizer_s2pro.py``.

    # Ronda 9 — Iter 3: vuelve a 1.05 (baseline). Iter 2 (1.02) no
    # redujo los saltos de timbre — confirmando que son intrínsecos
    # a XTTS y no dependen del pace. Mantenemos 1.05 por la fluidez
    # percibida del baseline (ronda 8 logró MOS 1.5 con este valor).
    tts_speed: float = 1.05
    # 0.75 — RONDA 10: revertido desde 0.60 al DEFAULT oficial de XTTS.
    # Las rondas 1-7 bajaron este valor persiguiendo la alucinación a chino
    # pero la causa real era el code-switching (fixed R8) + un
    # repetition_penalty absurdamente bajo (corregido en R10). Con 0.60 el
    # sampler tiene pocas opciones viables y cuando el token ideal no
    # encaja se encadena a tokens incoherentes → balbuceo que el usuario
    # reportó en E02/E03. El default 0.75 es el probado por Coqui.
    tts_temperature: float = 0.75
    target_language: str = "es"
    # Per-chunk char budget. 180 (antes 260) se alinea con merge_max_chars
    # 160: si el splitter interno es más agresivo que el merge, ninguna
    # frase fusionada llegará a la zona de contexto donde XTTS alucina
    # o trunca el final. Inputs más largos se parten en puntuación por
    # ``_split_long_text``.
    tts_char_limit: int = 180
    # Crossfade inside a phrase (between XTTS chunks of a long line).
    tts_crossfade_ms: int = 120
    # Crossfades between adjacent phrases (different TTS renders). The
    # inter-phrase fade masks the small timbre/level discontinuity that
    # remains after global level normalization. ``force_crossfade_ms``
    # applies to boundaries the overlap resolver had to touch.
    # 150 ms (antes 100) — cruces naturales entre frases cercanas más
    # suaves, disimulan el salto de RMS residual tras RMS-norm.
    # Ronda 9 Iter 3 probó 200 ms y empeoró hard_cuts (11→19) — el
    # fade largo convierte warnings en hard_cuts porque expone más
    # boundaries al ventaneo RMS/spectral del QA. 150 ms queda.
    inter_phrase_crossfade_ms: int = 150
    inter_phrase_crossfade_max_gap_ms: int = 300
    # 320 ms (antes 250) — ronda 9: los saltos de timbre (centroid
    # jump > 1200 Hz) son intrínsecos a XTTS (cada render tiene
    # espectro propio). No podemos eliminarlos normalizando, pero un
    # crossfade más largo los enmascara perceptivamente. Cap al 20%
    # de la frase en mixer impide que este valor deforme frases
    # cortas (una frase de 1.5 s queda capeada a 300 ms de fade,
    # ok en la práctica). 350+ ms empieza a "tragarse" la cabeza de
    # la frase siguiente en material rápido. Ronda 9 Iter 3 probó
    # 400 ms y empeoró hard_cuts — fades excesivos exponen más
    # ventanas al QA.
    force_crossfade_ms: int = 320
    # Tier 3 crossfade used when the pipeline's pairwise RMS leveling
    # pass flags a boundary as ``rms_jump_boundary``. Longer than
    # ``force_crossfade_ms`` because the boundary had a >6 dB delta even
    # after per-segment RMS normalization. Set per-engine in app.py
    # (ElevenLabs=550, XTTS keeps the default). None/0 = disabled.
    rms_jump_crossfade_ms: int = 0
    # 5.0 — RONDA 10: revertido desde 2.0. El default XTTS es 10.0.
    # Rondas 1-7 pensaban "más alto = más penaliza repetición" y bajaron
    # a 1.45 → 2.0 creyendo que endurecían, pero el efecto real fue
    # PERMITIR repeticiones → el sampler se atascaba en bucles de
    # tokens que luego colapsaban en balbuceo/chino. 5.0 es compromiso
    # entre el default 10.0 (puede sonar algo entrecortado en ES) y el
    # 2.0 roto anterior.
    tts_repetition_penalty: float = 5.0
    # 0.85 — RONDA 10: revertido a default XTTS. 0.80 cortaba el nucleus
    # demasiado agresivamente en ES, donde los fonemas vocálicos largos
    # necesitan más variabilidad para sonar naturales. Combinado con
    # repetition_penalty 2.0 y temperature 0.60 producía el balbuceo
    # reportado en E02/E03.
    tts_top_p: float = 0.85

    # Engine selector. Default is 's2pro' (local fish-speech via s2.cpp).
    # 'elevenlabs' / 'piper' / 'kokoro' are alternative active engines;
    # 'xtts' fields below are vestigial — kept as dataclass defaults so
    # external callers don't break, but no synthesizer reads them.
    tts_engine: str = "s2pro"
    xtts_model_name: str = ""
    xtts_config_path: str = ""
    xtts_checkpoint_dir: str = ""
    xtts_use_deepspeed: bool = False
    # ES/EN code-switching: DESACTIVADO por defecto (ronda 8 — fix definitivo
    # de alucinación a chino/japonés).
    #
    # Historial: durante rondas 1-7 esto estuvo ``True`` con un pipeline que
    # partía cada línea SRT en spans ES/EN y llamaba ``inference()`` por chunk
    # con ``language`` distinta. La idea era que "underhook" sonase con
    # fonología inglesa. El problema: XTTS-v2 **no está diseñado para
    # code-switching** — su parámetro ``language`` es single-value por
    # inferencia y el GPT-autoregresivo arrastra estado entre llamadas. Al
    # alternar es→en→es dentro de un mismo discurso, el sampler deja tokens
    # residuales del idioma anterior; como XTTS soporta 16 idiomas
    # (incluyendo zh-cn y ja) el decoder a veces colapsa a esos embeddings
    # vecinos → alucinación a chino/japonés. Evidencia comunidad:
    # coqui-ai/TTS discussions #4146, #104 en HF.
    #
    # Rondas 1-7 intentaron mitigarlo con: castellanización de compuestos
    # (bjj_casting.py), reducción del set 1-palabra (bjj_en_terms.py),
    # endurecimiento del sampler (temperature 0.60, top_p 0.80,
    # repetition_penalty 2.0), cap de chunk (180 chars). Todas ayudaron
    # pero ninguna eliminó al 100%. La ronda 7 QA seguía mostrando
    # alucinación residual en términos 1-palabra aislados (underhook,
    # guard, hook, kimura…).
    #
    # Fix definitivo: TODO el texto se sintetiza como single-span
    # ``language="es"``. Los términos BJJ que queden escritos en EN
    # (porque no estén en el mapa de castellanización) se pronuncian
    # con fonemas ES. "kimura" suena ES-nativo igual; "underhook" suena
    # "underjok" pero el oyente BJJ-literate lo reconoce igual. Esto es
    # MUCHO más aceptable que cualquier alucinación. Cumple el
    # invariante #4 del usuario ("NO alucinar") que tiene prioridad
    # absoluta sobre fidelidad fonética marginal.
    #
    # ¡NO REACTIVAR SIN LEER ESTO! Si en el futuro aparece un XTTS-v3
    # con code-switching real, o se migra a otro motor (CosyVoice, F5,
    # Higgs), entonces reabrir el debate. Hasta entonces, ``False`` es
    # la única configuración que garantiza cero alucinación.
    xtts_code_switching: bool = False
    xtts_en_terms_extra: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # ElevenLabs (alternative cloud backend)
    # ------------------------------------------------------------------
    # Swap by setting ``tts_engine = "elevenlabs"``. API key is read from
    # ``ELEVENLABS_API_KEY`` env var; voice_id is the pre-cloned speaker
    # registered in the ElevenLabs dashboard (PVC / IVC). Unlike XTTS,
    # reference_wav is ignored — the cloned voice lives on the provider.
    elevenlabs_voice_id: str = "LlZr3QuzbW4WrPjgATHG"
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    elevenlabs_stability: float = 0.5
    elevenlabs_similarity_boost: float = 0.75
    elevenlabs_style: float = 0.0
    elevenlabs_use_speaker_boost: bool = True
    # Output at 24 kHz PCM to match XTTS sample rate → the rest of the
    # audio pipeline (mixer, stretcher, demucs) needs no change.
    elevenlabs_output_format: str = "pcm_24000"
    elevenlabs_api_key_env: str = "ELEVENLABS_API_KEY"
    # HTTP timeout per request (seconds). Phrases are short so 60 s is
    # generous; dubbing 2000-char episodes splits into ~30-50 calls.
    elevenlabs_request_timeout: float = 60.0

    # ------------------------------------------------------------------
    # Piper (local ONNX TTS, free, no cloning)
    # ------------------------------------------------------------------
    # Spanish male voice from rhasspy/piper-voices. The model is baked
    # into the dubbing-generator image at build time. ``length_scale``
    # controls cadence (1.0 = native pace, >1 slower, <1 faster).
    # ``noise_scale`` and ``noise_w`` control prosodic variability.
    #
    # Defaults tuneados para narración BJJ:
    # - length_scale 0.95: cadencia ~5% más rápida que el default. El
    #   default 1.0 mete pausas demasiado largas; <0.9 sonaba apresurado
    #   y robótico.
    # - noise_scale 0.75 (antes 0.667): más variación tonal — el default
    #   sonaba "gangoso" en Piper-sharvard porque la prosodia era muy
    #   plana. Subirlo aporta naturalidad orgánica al precio de leve
    #   inestabilidad ocasional (aceptable en narración).
    # - noise_w 0.85 (antes 0.6): variación de duración fonética cercana
    #   al default Piper. Bajarlo demasiado homogeniza las pausas entre
    #   palabras y suena robótico.
    piper_model_path: str = "/models/piper/es_ES-sharvard-medium.onnx"
    piper_length_scale: float = 0.95
    piper_noise_scale: float = 0.75
    piper_noise_w: float = 0.85

    # ------------------------------------------------------------------
    # Kokoro-82M (local StyleTTS2, voz preset ES, gratis, mejor prosodia)
    # ------------------------------------------------------------------
    # ``em_alex`` y ``em_santa`` son las dos voces masculinas ES nativas.
    # Output 24 kHz mono nativo (no requiere resample).
    # speed=1.0 default; <1 más lento, >1 más rápido.
    kokoro_lang_code: str = "e"           # Spanish
    kokoro_voice: str = "em_alex"
    kokoro_speed: float = 1.0

    # ------------------------------------------------------------------
    # Fish Audio S2-Pro (local CUDA voice-clone TTS)
    # ------------------------------------------------------------------
    # s2.cpp inference engine running as HTTP server inside the container.
    # The server is booted at FastAPI startup (s2pro_server_manager) and
    # stays resident — model load takes ~10 s and we don't want to pay it
    # per phrase.
    #
    # Backend: CUDA (no Vulkan). Vulkan was discarded because Docker
    # Desktop on Windows/WSL2 doesn't expose the NVIDIA Vulkan ICD into
    # containers — only CUDA passes through. CUDA also gives us
    # dev/prod parity (workstation WSL2 + LXC Proxmox both use CUDA).
    #
    # IMPORTANT: ``s2_ref_text`` MUST match what the speaker says in the
    # ``s2_ref_audio_path`` WAV exactly. Drift between the two collapses
    # voice-clone quality (the model conditions on aligned phoneme→codec
    # pairs). If you swap the ref WAV, swap this string too.
    s2_server_host: str = "127.0.0.1"
    s2_server_port: int = 3030
    # GGUF + tokenizer paths are env-overridable so the operator can swap
    # quantizations (Q6_K → Q4_K_M for 6 GB VRAM cards) without rebuilding
    # the image. Compose passes S2PRO_GGUF_PATH / S2PRO_TOKENIZER_PATH.
    s2_gguf_path: str = field(
        default_factory=lambda: _env("S2PRO_GGUF_PATH", "/models/s2pro/s2-pro-q6_k.gguf")
    )
    s2_tokenizer_path: str = field(
        default_factory=lambda: _env("S2PRO_TOKENIZER_PATH", "/models/s2pro/tokenizer.json")
    )
    s2_ref_audio_path: str = "/voices/voice_martin_osborne_24k.wav"
    s2_ref_text: str = (
        "nunca te olvidé, nunca, el último beso que me diste todavía está "
        "grabado en mi corazón, por el día todo es más fácil. pero, todavía "
        "sueño contigo."
    )
    # Sampling: fish-speech upstream defaults. Don't lower these blindly —
    # narrow sampling on Spanish flattens prosody. 0.8/0.8/30 was validated
    # on 5 BJJ-ES smoke phrases on 2026-04-27.
    s2_temperature: float = 0.8
    s2_top_p: float = 0.8
    s2_top_k: int = 30
    s2_max_tokens: int = 1024
    # Per-phrase HTTP timeout. RTX 2060 + Vulkan: ~6 s audio in ~120 s.
    # 180 s leaves room for the longest merged phrase (~160 chars).
    s2_request_timeout: float = 180.0
    # Server boot health timeout. GGUF mmap from NFS-backed /models/s2pro
    # can take 20+ s on cold cache; 60 s is generous.
    s2_health_timeout_s: float = 60.0
    s2_cuda_device: int = 0

    # ------------------------------------------------------------------
    # Voice cloning
    # ------------------------------------------------------------------
    # XTTS needs 6-15 s of clean reference speech. More hurts: extra
    # duration dilutes timbre and picks up disfluencies from the clip.
    voice_sample_duration: float = 12.0
    use_model_voice: bool = False
    model_voice_path: str = ""

    # ------------------------------------------------------------------
    # Time-stretching (fit TTS into SRT slots)
    # ------------------------------------------------------------------
    # ES is usually 15-30% longer than EN. El traductor presupuesta la
    # mayor parte; el stretch absorbe el residuo. 1.25 (antes 1.30):
    # bajamos el techo base para evitar que todas las frases suenen
    # aceleradas — solo las del tramo final con deriva acumulada suben
    # a ``tail_speed_nudge_max_ratio=1.35`` vía el nudge. Hasta 1.30
    # es aceptable auditivamente; por encima de 1.35 la voz empieza
    # a sonar acelerada y el stretcher pitch-preserving introduce
    # artefactos F0 (saltos de pitch 100+ Hz en la QA).
    max_compression_ratio: float = 1.25
    min_compression_ratio: float = 0.90
    silence_threshold_db: float = -40.0
    silence_chunk_ms: int = 10

    # ------------------------------------------------------------------
    # Drift correction (density-based speed nudging)
    # ------------------------------------------------------------------
    # Off by default: with XTTS the natural pace already fits the slots
    # most of the time, and dynamic speed changes between phrases were
    # the main source of the "accelerates/brakes mid-chapter" artefact.
    drift_check_interval: int = 8
    drift_threshold_ms: float = 250.0
    speed_base: float = 1.0
    speed_min: float = 0.95
    speed_max: float = 1.10
    constant_speed: bool = True

    # ------------------------------------------------------------------
    # Ducking (voice over background)
    # ------------------------------------------------------------------
    ducking_bg_volume: float = 0.12    # ~-18 dB
    ducking_fg_volume: float = 1.6     # +4 dB TTS
    ducking_fade_ms: int = 180
    # Sustain the duck across short inter-phrase gaps so the background
    # doesn't pump up/down between consecutive coach utterances.
    ducking_sustain_gap_ms: int = 1500

    # ------------------------------------------------------------------
    # Sync / alignment
    # ------------------------------------------------------------------
    lookahead_phrases: int = 5
    min_phrase_duration_ms: int = 600
    avg_ms_per_char: float = 65.0       # Spanish ≈ 65 ms/char
    inter_phrase_pad_ms: int = 20       # minimum breathing room between phrases
    tts_lead_silence_ms: int = 0        # no leading silence — prosody links with prev tail

    # Industry-standard iso-synchrony. 250 ms es un techo pequeño que
    # obliga a la tubería a estirar (compression) antes que acumular
    # overflow. Con 600 ms las últimas frases arrastraban colas que al
    # final del vídeo no tenían room para empujar → solapes masivos.
    max_overflow_ms: int = 250
    shift_subsequent_on_overflow: bool = False

    # Compact-trailing-silence: off with SRT-anchored input; the gaps are
    # real coach pauses that must be preserved to keep lip sync.
    compact_trailing_silence: bool = False
    compact_trailing_silence_threshold_ms: int = 250
    compact_min_gap_ms: int = 80

    # Synthetic-silence compactor (words.json guided): when the TTS
    # render finishes well before the next anchor and words.json shows
    # the speaker was still talking in that gap, we pull the next phrase
    # back. If words.json reports real silence (camera cut, long breath),
    # the gap is preserved so lip sync doesn't drift.
    compact_synthetic_silence: bool = True
    compact_synthetic_silence_threshold_ms: int = 600
    compact_synthetic_min_speech_ratio: float = 0.70
    compact_synthetic_min_gap_ms: int = 200
    # Lip-sync safety rail: no phrase can be dragged more than this
    # before its original anchor. Protects the speaker's face from going
    # out of sync when compactor passes chain up.
    compact_synthetic_max_drift_ms: int = 400
    # Tail extension: permitimos que el audio final exceda la duración
    # del vídeo hasta ``video_tail_extension_max_ms`` para no perder la
    # última frase. Bajado de 8000 → 3000 ms tras ver que con 8s el
    # usuario notaba 8s de deriva al final del capítulo (el pipeline
    # usaba TODO el margen disponible). 3000 ms cubre la última frase
    # ES larga típica (~150 chars) sin producir deriva perceptible.
    # El nudge de speed en el tramo final (``tail_speed_nudge_*``)
    # absorbe el exceso cuando 3 s no bastarían.
    allow_video_tail_extension: bool = True
    # R11: 6000 ms. El tail=1500 (R10.2) truncaba contenido: el usuario
    # confirmó "acabó en bloque 30 pero quedaba 31 por decir". Con 6000
    # ms el último bloque SRT cabe completo incluso si dura 4-6 s más
    # que su slot original. El nudge ataca el problema desde 30 s antes
    # para que la mayoría de frases finales NO necesiten este margen —
    # 6 s es un backstop, no el objetivo. Usuario prioriza no perder
    # información > deriva visual corta pantalla negra.
    video_tail_extension_max_ms: int = 6000

    # Anti-deriva final: cuando las frases caen en el último
    # ``tail_speed_nudge_window_ms`` del vídeo y la posición donde
    # termina la frase excede el slot SRT por más de
    # ``tail_speed_nudge_trigger_ms``, aceleramos la compresión
    # hasta ``tail_speed_nudge_max_ratio`` para reabsorber el
    # retraso antes del final del vídeo. Sin esto, la única válvula
    # era ``video_tail_extension`` → el vídeo se alargaba en vez de
    # comprimir.
    #
    # 1.35 (antes 1.40): a 1.40x el stretcher pitch-preserving
    # introducía artefactos F0 perceptibles (QA reportaba saltos
    # de pitch 100+ Hz en las últimas frases). 1.35 sigue permitiendo
    # reabsorber ~800 ms de deriva por frase sin llegar a la zona
    # donde el pitch se pierde. Base 1.25 + nudge 1.35 = el nudge
    # sólo sube 10% sobre el ratio base, imperceptible.
    # R11: nudge aún más agresivo — 45 s de ventana para prevenir
    # acumulación, trigger a 400 ms (reacciona antes), max_ratio 1.45
    # (el stretch pitch-preserving sigue aceptable hasta 1.50). La
    # mayoría de frases finales deberían caber con stretch 1.0-1.15;
    # solo cuando se acumula mucha deriva sube a 1.45 y el oyente
    # percibirá aceleración leve en las últimas 3-5 frases — mejor
    # eso que perder contenido o tener 6 s de pantalla negra.
    tail_speed_nudge_window_ms: int = 45000
    tail_speed_nudge_trigger_ms: int = 400
    tail_speed_nudge_max_ratio: float = 1.45

    # Merge híbrido: ACTIVO. El usuario reporta "frase tras frase sin
    # continuidad" — síntoma clásico de XTTS renderizando cada SRT block
    # como inferencia independiente (el GPT-latent reinicia prosodia
    # entre bloques). Uniendo pares/tríos con gap <= 400 ms en un solo
    # prompt, el modelo mantiene entonación continua y los límites
    # resultantes son mucho menos "cortados".
    #
    # Cap 300 chars (s2pro-tuned). The XTTS-era cap was 160 to dodge
    # cross-language hallucination inside long merges, irrelevant for
    # s2.cpp. 300 keeps phrases under the 800-token degradation threshold
    # while collapsing more SRT splits into single prompts (~25-40%
    # fewer renders per episode) — net reduction in boundary count and
    # smoother prosody. ElevenLabs overrides to 500 in app.py.
    merge_consecutive_blocks: bool = True
    merge_max_gap_ms: int = 400
    merge_max_chars: int = 300

    # ------------------------------------------------------------------
    # Paths / file discovery
    # ------------------------------------------------------------------
    extensions: tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov")
