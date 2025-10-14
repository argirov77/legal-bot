"""Application package initialisation."""

# Import the NumPy compatibility shims so that they are applied before any
# third-party modules (such as ChromaDB) are imported. The imported name is not
# used directly, but the module's side effects are required.
from . import _numpy_compat as _numpy_compat  # noqa: F401

