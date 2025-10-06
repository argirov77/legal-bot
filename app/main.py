"""Compatibility wrapper for :mod:`src.app.main`.

Uvicorn expects to import ``app.main:app`` based on the project README
and container configuration.  When the source layout keeps the real
application inside ``src/app``, importing ``app`` would normally fail.
This module re-exports the FastAPI ``app`` instance from its actual
location so that existing commands continue to work.
"""
from src.app.main import app

__all__ = ["app"]
