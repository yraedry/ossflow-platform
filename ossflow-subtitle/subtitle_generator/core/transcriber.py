"""Bridge ``RunRequest`` → ``SubtitlePipeline``.

Migrado de ``app.py`` en T31.5. Recibe el ``RunRequest`` del kit y
ejecuta el flujo:

* Si ``opts.translate_only=True`` delega a ``translate_runner``.
* Si no, configura prompts/CUDA y lanza ``SubtitlePipeline.process_*``.

Cleanup GPU garantizado en ``finally`` para no leakear VRAM entre jobs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ossflow_service_kit import JobEvent, RunRequest, emit_logs

from subtitle_generator.shared.paths import resolve_input


def run_subtitle_generator(req: RunRequest, emit) -> None:
    """Bridge ``RunRequest`` → ``SubtitlePipeline``.

    Cuando ``options.translate_only=True`` corre SRT translation
    (EN→ES vía Ollama/OpenAI) en vez de transcripción —
    reutilizando el mismo contrato job/SSE.
    """
    opts = req.options or {}

    if opts.get("translate_only"):
        from subtitle_generator.core.translate_runner import (
            run_translate_directory,
        )
        run_translate_directory(req, emit)
        return

    input_path = resolve_input(Path(req.input_path))

    level = logging.DEBUG if opts.get("verbose") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    with emit_logs(emit, level=level):
        from subtitle_generator.config import (
            DEFAULT_INITIAL_PROMPT,
            SubtitleConfig,
            TranscriptionConfig,
            generate_prompt,
        )
        from subtitle_generator.cuda_setup import (
            setup_nvidia_dlls,
            setup_pytorch_safety,
        )
        from subtitle_generator.pipeline import SubtitlePipeline

        setup_nvidia_dlls()
        setup_pytorch_safety()

        if opts.get("prompt") is not None:
            initial_prompt = opts["prompt"]
        elif opts.get("instructor") or opts.get("topic"):
            initial_prompt = generate_prompt(
                instructor=opts.get("instructor"),
                topic=opts.get("topic"),
            )
        else:
            initial_prompt = DEFAULT_INITIAL_PROMPT

        t_config = TranscriptionConfig(
            model_name=opts.get("model", "large-v3"),
            language=opts.get("language", "en"),
            batch_size=int(opts.get("batch_size", 4)),
            initial_prompt=initial_prompt,
            postprocess_openai=bool(opts.get("postprocess_openai", False)),
            postprocess_model=str(opts.get("postprocess_model", "gpt-4o-mini")),
            postprocess_api_key=(
                opts.get("postprocess_api_key")
                or os.environ.get("OPENAI_API_KEY")
            ),
        )
        s_config = SubtitleConfig()

        force = bool(opts.get("force", False))
        emit(JobEvent(type="log", data={"message":
            f"starting subtitle-generator on {input_path}"
            + (" (force overwrite)" if force else "")
        }))
        import gc

        import torch

        pipeline = SubtitlePipeline(t_config, s_config)
        pipeline.load_models()
        try:
            if input_path.is_file():
                pipeline.process_file(input_path, force=force)
            else:
                pipeline.process_directory(input_path, force=force)
            emit(JobEvent(type="progress", data={"pct": 100}))
        finally:
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
