"""CLI entry point for running the subtitle generator as a package via `python -m subtitle_generator`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_ROOT_DIR, DEFAULT_INITIAL_PROMPT, TranscriptionConfig, SubtitleConfig, generate_prompt
from .cuda_setup import setup_nvidia_dlls, setup_pytorch_safety
from .pipeline import SubtitlePipeline

log = logging.getLogger("subtitler")


def main() -> None:
    """Parse arguments and run the subtitle pipeline."""
    parser = argparse.ArgumentParser(
        description="Transcribe video audio to SRT subtitles using WhisperX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m subtitle_generator "Z:\\path\\to\\videos"\n'
            '  python -m subtitle_generator "Z:\\path" --model large-v3 --verbose\n'
            '  python -m subtitle_generator "Z:\\path" --batch-size 8 --prompt "Custom prompt"\n'
        ),
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=DEFAULT_ROOT_DIR,
        help=f"Root directory to scan for video files (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument("--model", default="large-v3", help="Whisper model name (default: large-v3)")
    parser.add_argument("--language", default="en", help="Audio language code (default: en)")
    parser.add_argument("--prompt", default=None, help="Initial prompt for domain vocabulary")
    parser.add_argument("--instructor", default=None, help="Instructor name for dynamic prompt generation")
    parser.add_argument("--topic", default=None, help="Topic/title for dynamic prompt generation")
    parser.add_argument("--batch-size", type=int, default=4, help="Transcription batch size (default: 4)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    root_dir = Path(args.directory)
    if not root_dir.exists():
        log.error("Directory does not exist: %s", root_dir)
        sys.exit(1)

    # Setup CUDA environment
    setup_nvidia_dlls()
    setup_pytorch_safety()

    # Build initial prompt: explicit --prompt wins, then --instructor/--topic, then generic
    if args.prompt is not None:
        initial_prompt = args.prompt
    elif args.instructor or args.topic:
        initial_prompt = generate_prompt(instructor=args.instructor, topic=args.topic)
    else:
        initial_prompt = DEFAULT_INITIAL_PROMPT

    # Build configs
    t_config = TranscriptionConfig(
        model_name=args.model,
        language=args.language,
        batch_size=args.batch_size,
        initial_prompt=initial_prompt,
    )
    s_config = SubtitleConfig()

    # Run pipeline
    pipeline = SubtitlePipeline(t_config, s_config)
    try:
        pipeline.load_models()
    except Exception as e:
        log.error("Failed to load models: %s", e, exc_info=True)
        sys.exit(1)

    pipeline.process_directory(root_dir)


if __name__ == "__main__":
    main()
