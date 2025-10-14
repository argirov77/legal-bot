"""Pytest configuration for lightweight dependency shims."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:  # pragma: no branch - deterministic path
    sys.path.insert(0, str(SRC_DIR))

_SITE_CUSTOMIZE = ROOT_DIR / "sitecustomize.py"
if _SITE_CUSTOMIZE.exists():  # pragma: no branch - deterministic path
    spec = importlib.util.spec_from_file_location("sitecustomize", _SITE_CUSTOMIZE)
    if spec and spec.loader:  # pragma: no branch - defensive
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("sitecustomize", module)
        spec.loader.exec_module(module)
