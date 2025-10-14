"""Test-time compatibility patches applied automatically by Python."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import ForwardRef

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:  # pragma: no branch - deterministic path
    sys.path.insert(0, str(_SRC))

# Ensure Pydantic v1 remains compatible with Python 3.12's ForwardRef changes.
if "recursive_guard" in inspect.signature(ForwardRef._evaluate).parameters:  # pragma: no branch
    _original_evaluate = ForwardRef._evaluate

    def _compat_forwardref_evaluate(self, globalns, localns, *args, **kwargs):  # type: ignore[override]
        recursive_guard = kwargs.pop("recursive_guard", None)
        type_params = None

        if args:
            if len(args) == 1:
                if recursive_guard is None:
                    recursive_guard = args[0]
                else:
                    type_params = args[0]
            else:
                type_params = args[0]
                recursive_guard = args[1]

        if recursive_guard is None:
            recursive_guard = set()

        return _original_evaluate(self, globalns, localns, type_params, recursive_guard=recursive_guard)

    ForwardRef._evaluate = _compat_forwardref_evaluate  # type: ignore[assignment]

