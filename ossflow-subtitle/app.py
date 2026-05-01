"""FastAPI entrypoint del servicio subtitle-generator (T31.8).

Tras el Plan 3 T31, este archivo es solo el punto de entrada del
servicio. La lógica vive en ``subtitle_generator/`` (paquete) bajo
``api/``, ``core/`` y ``shared/``.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_KIT_PARENT = Path(__file__).resolve().parent.parent
if str(_KIT_PARENT) not in sys.path:
    sys.path.insert(0, str(_KIT_PARENT))

from ossflow_service_kit import create_app  # noqa: E402

from subtitle_generator.api import router as _router_mod  # noqa: E402
from subtitle_generator.core.transcriber import (  # noqa: E402
    run_subtitle_generator,
)
from subtitle_generator.shared.hf_cache import clear_hf_locks  # noqa: E402

SERVICE_NAME = "subtitle-generator"


# Best-effort cleanup at startup — frees locks left by a killed worker.
try:
    _startup_clean = clear_hf_locks()
    if _startup_clean["removed"]:
        logging.getLogger(SERVICE_NAME).warning(
            "cleared %d stale HF lock(s) at startup",
            _startup_clean["removed"],
        )
except Exception as _exc:
    logging.getLogger(SERVICE_NAME).warning(
        "HF lock cleanup at startup failed: %s", _exc,
    )


app = create_app(service_name=SERVICE_NAME, task_fn=run_subtitle_generator)
_router_mod.register(app)


# ---------------------------------------------------------------------------
# Compat re-exports (para tests que parchean ``app.X`` por nombre).
# ---------------------------------------------------------------------------

from subtitle_generator.api.router import (  # noqa: E402,F401
    AnalyzeRequest,
    ApplyRequest,
    RegenerateRequest,
    TranslateRequest,
    ValidateRequest,
)
from subtitle_generator.core.regenerator import (  # noqa: E402,F401
    get_regenerator as _get_regenerator,
)
from subtitle_generator.core.transcriber import (  # noqa: E402,F401
    run_subtitle_generator as _run_subtitle_generator,
)
from subtitle_generator.core.translate_runner import (  # noqa: E402,F401
    build_translator_with_fallback as _build_translator_with_fallback,
    run_translate_directory as _run_translate_directory,
    translate_for_dubbing as _translate_for_dubbing,
    translate_for_dubbing_nivel3 as _translate_for_dubbing_nivel3,
)
from subtitle_generator.shared.hf_cache import (  # noqa: E402,F401
    clear_hf_locks as _clear_hf_locks,
    hf_cache_root as _hf_cache_root,
)
from subtitle_generator.shared.paths import (  # noqa: E402,F401
    clean_base_stem as _clean_base_stem,
    dub_srt_path_for as _dub_srt_path_for,
    literal_srt_path_for as _literal_srt_path_for,
    resolve_input as _resolve_input,
    words_json_for as _words_json_for,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
