"""CLI entry point: python -m dubbing_generator <directory> [options]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DubbingConfig
from .pipeline import DubbingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dub BJJ instructional videos to Spanish with Coqui XTTS v2",
    )
    parser.add_argument(
        "directory",
        help="Root directory containing video files and SRT subtitles",
    )
    parser.add_argument(
        "--voice-profile",
        default=None,
        help="Path to a WAV file to use as voice reference (overrides extraction)",
    )
    parser.add_argument(
        "--use-model-voice",
        action="store_true",
        help="Use a pre-recorded model voice instead of cloning from the video",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("dubbing_generator")

    root_dir = Path(args.directory)
    if not root_dir.exists():
        logger.error("Directory does not exist: %s", root_dir)
        sys.exit(1)

    # Build config
    config = DubbingConfig(
        use_model_voice=args.use_model_voice,
    )

    # Build and run pipeline
    voice_ref = Path(args.voice_profile) if args.voice_profile else None

    pipeline = DubbingPipeline(config)
    results = pipeline.process_directory(root_dir)

    logger.info("Completed. %d videos dubbed.", len(results))


if __name__ == "__main__":
    main()
