"""Core del servicio dubbing-generator (Plan 3 T32).

Componentes extraídos del monolito ``pipeline.py`` (1469 LOC):

* ``srt_parser`` — parse SRT + helpers de tiempo + prosodic continuity.
* ``audio_normalizer`` — RMS normalize, gap compaction, overlap resolve.
* ``muxer`` — ffmpeg merge audio + video.
* ``qa_runner`` — boundary checks, MOS, TTS-only export, verdict.
* ``synthesizer`` — bucle TTS S2-Pro (extraído de ``_synthesize_all``).
"""
