"""Compatibility package exposing the FastAPI application.

This shim allows running ``uvicorn app.main:app`` without requiring
``src`` to be on the Python import path.  It simply re-exports the
application object from the actual implementation under ``src/app``.
"""
from src.app.main import app

__all__ = ["app"]
