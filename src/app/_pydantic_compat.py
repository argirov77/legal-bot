"""Compatibility helpers for running Pydantic v1 on Python 3.12+."""
from __future__ import annotations

import inspect
from typing import ForwardRef

# Python 3.12 changed the signature of ``ForwardRef._evaluate`` by adding a
# keyword-only ``recursive_guard`` argument. Pydantic v1 still calls the method
# using the legacy positional signature, which raises ``TypeError`` on modern
# interpreters. The shim below adapts the call signature back to the shape
# expected by Pydantic while delegating to the standard library implementation.
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

