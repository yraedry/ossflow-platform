"""CLI entry point for running the package via `python -m chapter_splitter`."""

import argparse

from .config import Config, DEFAULT_ROOT
from .pipeline import Pipeline
from .utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BJJ Instructional Video Chapter Splitter"
    )
    parser.add_argument(
        "root", nargs="?", default=DEFAULT_ROOT,
        help="Root directory to scan for videos (default: %(default)s)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Detect chapters but do not create files"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--voice-threshold", type=float, default=0.25,
        help="Legacy RMS threshold for voice detection (default: 0.25)"
    )
    parser.add_argument(
        "--voice-enter", type=float, default=0.30,
        help="Hysteresis enter threshold for voice detection (default: 0.30)"
    )
    parser.add_argument(
        "--voice-exit", type=float, default=0.20,
        help="Hysteresis exit threshold for voice detection (default: 0.20)"
    )
    parser.add_argument(
        "--scan-step", type=float, default=0.5,
        help="Scan interval in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--ocr-confidence", type=float, default=0.55,
        help="Minimum OCR confidence threshold (default: 0.55)"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    config = Config(
        root_dir=args.root,
        dry_run=args.dry_run,
        verbose=args.verbose,
        voice_threshold=args.voice_threshold,
        voice_enter_threshold=args.voice_enter,
        voice_exit_threshold=args.voice_exit,
        scan_step=args.scan_step,
        ocr_confidence_min=args.ocr_confidence,
    )

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
