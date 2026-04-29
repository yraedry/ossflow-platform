"""Density-aware speed corrector for TTS phrases."""

from __future__ import annotations

import logging

from ..config import DubbingConfig

logger = logging.getLogger(__name__)


class DriftCorrector:
    """Adjust TTS speed based on local phrase density.

    In anchor-based sync each phrase starts at its SRT timestamp, so there is
    no accumulated positional drift. Instead we look at text *density* (chars
    per allocated ms) and adjust TTS speed within a tight range so dense
    phrases get a slight speed-up (avoid 1.25x compression in stretcher) while
    sparse phrases slow down slightly (more natural delivery).
    """

    # Density reference: ~0.015 chars/ms ≈ 66 ms/char ≈ comfortable Spanish pace
    DENSITY_BASE = 0.015

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config
        self._current_speed: float = config.speed_base

    @property
    def current_speed(self) -> float:
        return self._current_speed

    def check_density(self, phrase_index: int, density: float) -> float:
        """Return TTS speed for a phrase based on text density.

        density = len(text) / allocated_ms. Higher density → speed up slightly.
        """
        # Ratio of this phrase density vs comfortable baseline
        pressure = density / self.DENSITY_BASE if self.DENSITY_BASE > 0 else 1.0

        # Soft mapping: pressure 1.0 → speed 1.0; pressure 1.3 → speed ~1.1
        # Keep within [speed_min, speed_max]
        target_speed = self.cfg.speed_base * (0.75 + 0.25 * pressure)
        target_speed = max(self.cfg.speed_min, min(self.cfg.speed_max, target_speed))

        # Smooth transitions (no jumps between consecutive phrases)
        self._current_speed = self._move_toward(
            self._current_speed, target_speed, step=0.04,
        )

        if phrase_index % self.cfg.drift_check_interval == 0:
            logger.debug(
                "Phrase %d: density=%.4f pressure=%.2f → speed=%.2f",
                phrase_index, density, pressure, self._current_speed,
            )

        return self._current_speed

    # Legacy positional API (kept for backward compatibility)
    def check(
        self,
        phrase_index: int,
        current_position_ms: int,
        expected_position_ms: int,
    ) -> float:
        """Legacy positional drift check. Returns current speed unchanged.

        Anchor-based sync makes positional drift irrelevant; kept only so
        external callers don't break.
        """
        return self._current_speed

    def reset(self) -> None:
        self._current_speed = self.cfg.speed_base

    @staticmethod
    def _move_toward(current: float, target: float, step: float) -> float:
        if abs(current - target) <= step:
            return target
        return current - step if current > target else current + step
