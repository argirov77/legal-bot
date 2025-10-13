"""Compatibility helpers for NumPy version differences.

This module provides shims for APIs that were removed in NumPy 2.0 but are
still referenced by some of our third-party dependencies (e.g. ChromaDB).
"""

from __future__ import annotations

import numpy as np


def apply_shims() -> None:
    """Restore deprecated NumPy aliases expected by older dependencies."""

    # ``np.float_`` was removed in NumPy 2.0. Some dependencies (such as
    # ChromaDB 0.4.x) still import it, so recreate the alias when necessary.
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]


# Apply the shims on import so that downstream modules immediately benefit.
apply_shims()

