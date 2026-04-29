"""Service-level FastAPI smoke tests for dubbing-generator."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

# Make the dubbing-generator dir importable as a package root for `app`.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def test_app_state_carries_s2pro_manager_after_startup(monkeypatch):
    """Startup hook puts the manager on app.state, not in a global."""
    monkeypatch.delenv("DUBBING_TTS_ENGINE", raising=False)
    import app as dub_app
    dub_app._start_s2pro_server()
    assert hasattr(dub_app.app.state, "s2pro_manager")


def test_s2pro_status_route_present():
    import app as dub_app
    routes = {r.path for r in dub_app.app.router.routes}
    assert "/s2pro/status" in routes


def test_run_dubbing_generator_accepts_s2pro_engine():
    import app as dub_app
    src = inspect.getsource(dub_app._run_dubbing_generator)
    assert '"s2pro"' in src
    # S2-Pro is the default engine since the migration completed (memory:
    # project_s2pro_integration). Env-fallback resolves to s2pro when
    # DUBBING_TTS_ENGINE is unset.
    assert 'or "s2pro"' in src
