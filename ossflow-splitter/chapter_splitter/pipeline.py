"""Top-level pipeline orchestrating voice analysis, OCR detection, and splitting."""

import logging
import os
import re
from pathlib import Path

import cv2

from .config import Config
from .utils import extract_season_number
from .ocr.reader import OcrReader
from .audio.analyzer import AudioAnalyzer
from .detection.detector import ChapterDetector
from .splitting.splitter import VideoSplitter

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the full processing workflow."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.ocr = OcrReader(config)
        self.audio = AudioAnalyzer(config)
        self.detector = ChapterDetector(config, self.ocr, self.audio)
        self.splitter = VideoSplitter(config)

    def process_video(self, video_path: Path, show_name: str,
                      season: int) -> None:
        """Process a single video file."""
        logger.info("=" * 60)
        logger.info("Processing: %s", video_path.name)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        if duration < 10:
            logger.warning("Video too short (%.1fs), skipping.", duration)
            cap.release()
            return

        logger.info("Duration: %.0fs (%.1f min), FPS: %.1f",
                     duration, duration / 60, fps)

        try:
            # Phase 1: pre-compute voice map
            self.audio.precompute(str(video_path), duration)

            # Phase 2: detect chapters
            chapters = self.detector.detect(str(video_path), cap, duration)

            # Phase 3: split
            output_dir = video_path.parent / f"Season {season:02d}"
            self.splitter.split(str(video_path), chapters, output_dir,
                                show_name, season, dry_run=self.cfg.dry_run)
        finally:
            cap.release()

    def run(self) -> None:
        """Walk the root directory and process all videos."""
        root = Path(self.cfg.root_dir)
        if not root.exists():
            logger.error("Root directory does not exist: %s", root)
            return

        logger.info("Starting pipeline on: %s", root)

        for dirpath, _dirnames, filenames in os.walk(root):
            videos = sorted(
                f for f in filenames
                if f.lower().endswith(self.cfg.extensions)
            )
            if not videos:
                continue

            folder_name = Path(dirpath).name
            if "season" in folder_name.lower():
                show_name = Path(dirpath).parent.name
            else:
                show_name = folder_name

            for file_idx, video_file in enumerate(videos, start=1):
                # Skip already-split outputs
                if "Technique" in video_file or re.search(r"S\d{2}E\d{2}", video_file):
                    continue

                video_path = Path(dirpath) / video_file
                season = extract_season_number(video_file, file_idx)
                self.process_video(video_path, show_name, season)

        logger.info("Pipeline complete.")
