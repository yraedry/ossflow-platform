"""Configuration dataclass and default constants for the chapter splitter."""

from dataclasses import dataclass

DEFAULT_ROOT = r"Z:\instruccionales\Inside Camping - Gordon ryan"


@dataclass
class Config:
    """All tuneable parameters in one place."""

    root_dir: str = DEFAULT_ROOT
    extensions: tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov")
    dry_run: bool = False
    verbose: bool = False

    # Audio / voice
    voice_threshold: float = 0.25  # legacy fallback (used if hysteresis disabled)
    voice_enter_threshold: float = 0.30  # RMS above which we declare voice present
    voice_exit_threshold: float = 0.20   # RMS below which we declare silence
    voice_window_ms: int = 100           # analysis window size in milliseconds
    voice_smoothing_ms: int = 500        # moving-average smoothing window in ms
    audio_chunk_seconds: float = 10.0

    # OCR
    ocr_confidence_min: float = 0.55
    ocr_title_confidence_min: float = 0.75  # higher bar to accept as CHAPTER TITLE
    ocr_min_text_length: int = 5         # ignore non-numeric texts shorter than this
    ocr_min_title_chars: int = 4         # minimum chars (post-strip) to accept as title
    ocr_voting_frames: int = 5
    ocr_voting_window: float = 1.0  # seconds over which voting frames are sampled
    max_title_length: int = 150

    # OCR blacklist (case-insensitive substring match on normalized text)
    ocr_blacklist: tuple[str, ...] = ("bjjfanatics", "fanatics", "bjj fanatics")

    # Watermark mask (fractional coords, (x0, y0, x1, y1) in 0..1).
    # Default cubre abajo-derecha 25% ancho × 20% alto (watermark BJJFanatics).
    watermark_mask_enabled: bool = True
    watermark_region: tuple[float, float, float, float] = (0.75, 0.80, 1.00, 1.00)

    # ROI
    roi_top_fraction: float = 0.15
    roi_bottom_fraction: float = 0.15

    # Background memory
    background_similarity: float = 0.45
    background_max_entries: int = 30
    background_boost_threshold: float = 0.8
    background_decay: float = 0.95

    # Stability
    stability_frames: int = 3
    stability_interval: float = 1.0  # seconds between stability check frames
    stability_agreement: float = 0.60
    stability_bidirectional: bool = True  # verify frames before AND after detection

    # Detection
    scan_step: float = 0.5
    cooldown_seconds: float = 5.0
    min_chapter_duration: float = 15.0
    min_chapter_duration_sec: float = 60.0   # hard floor; shorter chapters get merged
    dedupe_similarity_threshold: float = 0.25  # Levenshtein distance ratio threshold

    # Audio / color corroboration for transition decisions
    audio_corroboration: bool = True
    color_corroboration: bool = True
    corroboration_boost_audio: float = 0.10
    corroboration_boost_color: float = 0.05

    # Cut-point search
    cut_search_window: float = 3.0  # seconds to search backward for scene cut
    cut_mse_threshold: float = 800.0  # MSE above which we declare a scene change

    # Encoding
    encoder: str = "h264_nvenc"
    preset: str = "p4"
    cq: int = 23
    audio_codec: str = "aac"
    output_format: str = ".mkv"
