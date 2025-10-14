"""Application package initialisation."""

# Import the compatibility shims so that they are applied before any
# third-party modules (such as ChromaDB or FastAPI) are imported. The imported
# names are not used directly, but the module's side effects are required.
from . import _numpy_compat as _numpy_compat  # noqa: F401
from . import _pydantic_compat as _pydantic_compat  # noqa: F401

