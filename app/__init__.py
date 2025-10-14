"""Compatibility package exposing the FastAPI application and modules."""
import importlib
import sys

vectorstore_module = importlib.import_module("src.app.vectorstore")
sys.modules.setdefault("app.vectorstore", vectorstore_module)

ingest_module = importlib.import_module("src.app.ingest")
sys.modules.setdefault("app.ingest", ingest_module)

from src.app.main import app

__all__ = ["app"]
