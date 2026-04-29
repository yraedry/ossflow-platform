"""Tests for DriftCorrector (density-based speed adjustment).

Anchor-based sync hace que no haya drift posicional, así que probamos la
API activa (``check_density``) y verificamos que la API legacy
(``check``) mantiene compatibilidad pero ya no ajusta nada.
"""

from dubbing_generator.config import DubbingConfig
from dubbing_generator.sync.drift_corrector import DriftCorrector


def _make_corrector(**kwargs) -> DriftCorrector:
    cfg = DubbingConfig(**kwargs)
    return DriftCorrector(cfg)


class TestDriftCorrector:
    def test_reset_restores_base_speed(self):
        dc = _make_corrector()
        # Nudge density so speed moves away from base
        dc.check_density(0, density=0.025)
        dc.reset()
        assert dc.current_speed == dc.cfg.speed_base

    def test_density_above_base_speeds_up(self):
        dc = _make_corrector(speed_min=0.9, speed_max=1.2)
        # density 0.022 > DENSITY_BASE (0.015) → pressure > 1 → speed up
        speed = dc.check_density(0, density=0.022)
        assert speed > dc.cfg.speed_base

    def test_density_below_base_slows_down(self):
        dc = _make_corrector(speed_min=0.8, speed_max=1.2)
        # density 0.008 < DENSITY_BASE → pressure < 1 → slow down
        speed = dc.check_density(0, density=0.008)
        assert speed < dc.cfg.speed_base

    def test_speed_clamped_to_max(self):
        dc = _make_corrector(speed_max=1.10)
        # Massive density spike — target would exceed speed_max
        for i in range(50):  # step-by-step ramp
            speed = dc.check_density(i, density=0.1)
        assert speed <= 1.10

    def test_speed_clamped_to_min(self):
        dc = _make_corrector(speed_min=0.95)
        # Very low density — target would fall below speed_min
        for i in range(50):
            speed = dc.check_density(i, density=0.001)
        assert speed >= 0.95

    def test_legacy_check_returns_current_speed(self):
        """Legacy positional API is a no-op under anchor-based sync."""
        dc = _make_corrector()
        # Whatever arguments, it returns current_speed (== speed_base initially)
        speed = dc.check(
            0, current_position_ms=10500, expected_position_ms=10000,
        )
        assert speed == dc.cfg.speed_base
