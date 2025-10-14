"""Compatibility helpers for running Pydantic v1 on Python 3.12."""
from __future__ import annotations

import inspect
from typing import ForwardRef


def apply_shims() -> None:
    """Patch :func:`typing.ForwardRef._evaluate` signature changes in Python 3.12."""

    if not hasattr(ForwardRef, "_evaluate"):
        return

    signature = inspect.signature(ForwardRef._evaluate)
    parameter = signature.parameters.get("recursive_guard")
    if parameter is None:
        original = ForwardRef._evaluate

        def _patched_evaluate(self, globalns, localns, recursive_guard=None):  # type: ignore[override]
            if recursive_guard is None:
                recursive_guard = set()
            return original(self, globalns, localns)

        ForwardRef._evaluate = _patched_evaluate  # type: ignore[assignment]
        return

    if parameter.default is not inspect._empty:
        # Already backwards compatible.
        return

    original = ForwardRef._evaluate

    if "type_params" in signature.parameters:

        def _patched_evaluate(  # type: ignore[override]
            self,
            globalns,
            localns,
            type_params=None,
            *,
            recursive_guard=None,
        ):
            if recursive_guard is None:
                recursive_guard = set()
            return original(
                self,
                globalns,
                localns,
                type_params,
                recursive_guard=recursive_guard,
            )

    else:

        def _patched_evaluate(self, globalns, localns, recursive_guard=None):  # type: ignore[override]
            if recursive_guard is None:
                recursive_guard = set()
            return original(self, globalns, localns, recursive_guard)

    ForwardRef._evaluate = _patched_evaluate  # type: ignore[assignment]


apply_shims()
