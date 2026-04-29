"""Shared pytest config for processor-api tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure processor-api root is on sys.path so `import api.*` works.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default CONFIG_DIR to a scratch dir so module imports don't touch /data/config
os.environ.setdefault("CONFIG_DIR", str(ROOT / ".pytest_config"))
