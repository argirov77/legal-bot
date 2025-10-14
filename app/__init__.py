"""Compatibility package exposing the FastAPI application.

This shim allows running ``uvicorn app.main:app`` without requiring
``src`` to be on the Python import path.  Historically it simply
re-exported the application object from the implementation under
``src/app``.  As the project grew, more modules within ``src/app`` began
using absolute imports such as ``app.api``.  Without a corresponding
package structure under ``app`` those imports failed when the service
booted under Uvicorn.

To preserve the existing command-line interface while keeping the
``src`` layout, we dynamically extend the package search path so that the
``app`` namespace covers both the compatibility shim *and* the real
implementation directory.  This mirrors how namespace packages behave
and means ``import app.api`` works exactly as if the code lived directly
under ``app``.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_SRC_APP_DIR = _PACKAGE_DIR.parent / "src" / "app"

# ``__path__`` controls where Python looks for submodules of this
# package.  By appending ``src/app`` we make ``app.*`` imports resolve to
# the actual implementation modules.
if _SRC_APP_DIR.exists():
    __path__ = [str(_PACKAGE_DIR), str(_SRC_APP_DIR)]  # type: ignore[name-defined]


# Re-export the FastAPI application instance expected by ``uvicorn
# app.main:app``.
app = import_module("src.app.main").app

__all__ = ["app"]
