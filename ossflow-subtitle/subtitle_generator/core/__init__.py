"""Capacidades de dominio del servicio subtitle-generator (Plan 3 T31).

Migración en curso desde ``app.py`` (1040 LOC monolíticas) a
componentes pequeños testeables aislados. Anatomía objetivo:

* ``transcriber.py`` — wrapper WhisperX.
* ``translator.py`` — cliente Ollama/OpenAI (queda como ``subtitle_generator.translator``).
* ``translate_runner.py`` — orquestador batch del flujo translate.
* ``analyzer.py`` — analyze_video.
* ``regenerator.py`` — sustituye al global ``_regenerator`` (inyectable).
* ``postprocessor.py`` — limpieza OpenAI / hotwords.
* ``srt_io.py`` — parse/serialize SRT (ya existe en el package raíz).

Durante la migración los nuevos componentes coexisten con los archivos
flat originales en ``subtitle_generator/`` y ``app.py`` re-exporta para
mantener verde la suite de tests.
"""
