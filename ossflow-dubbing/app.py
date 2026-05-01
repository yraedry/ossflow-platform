"""FastAPI entrypoint del servicio dubbing-generator (T32.7+8).

Tras el Plan 3 T32, este archivo es solo el punto de entrada del
servicio. La lógica vive en ``dubbing_generator/`` (paquete) bajo
``api/``, ``core/`` y ``shared/``.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8003
"""

from __future__ import annotations

import sys
from pathlib import Path

_KIT_PARENT = Path(__file__).resolve().parent.parent
if str(_KIT_PARENT) not in sys.path:
    sys.path.insert(0, str(_KIT_PARENT))

from ossflow_service_kit import RunRequest, create_app  # noqa: E402

from dubbing_generator.api import router as _router_mod  # noqa: E402
from dubbing_generator.core.runner import (  # noqa: E402
    run_dubbing_generator as _core_run,
)


SERVICE_NAME = "dubbing-generator"


def _run_dubbing_generator(req: RunRequest, emit) -> None:
    """Bridge fino al runner del core, pasándole ``app.state`` para que
    ``/s2pro/status`` pueda consultar el manager activo."""
    _core_run(req, emit, app_state=app.state)


app = create_app(service_name=SERVICE_NAME, task_fn=_run_dubbing_generator)
_router_mod.register(app)


# ---------------------------------------------------------------------------
# Lifecycle: shutdown safety net para s2.cpp.
# ---------------------------------------------------------------------------
#
# Lazy-load: NO startup hook. El s2.cpp server lo arranca el runner solo
# durante un job s2pro y lo para en finally. El shutdown hook es safety
# net para procesos que mueran mid-job.

def _stop_s2pro_server() -> None:
    manager = getattr(app.state, "s2pro_manager", None)
    if manager is not None:
        manager.stop()


app.router.on_shutdown.append(_stop_s2pro_server)


# ---------------------------------------------------------------------------
# Compat re-exports (para tests/imports legacy de app.X).
# ---------------------------------------------------------------------------

from dubbing_generator.api.router import (  # noqa: E402,F401
    AnalyzeRequest,
)
from dubbing_generator.core.runner import (  # noqa: E402,F401
    resolve_input as _resolve_input,
    resolve_srt_for as _resolve_srt_for,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
