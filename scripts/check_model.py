#!/usr/bin/env python3
"""CLI helper that verifies whether the configured LLM can be loaded."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore import-not-found
    except Exception:  # pragma: no cover - optional dependency
        return

    project_root = Path(__file__).resolve().parents[1]
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:  # pragma: no cover - fallback path
        load_dotenv()


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> int:
    _load_dotenv()
    _configure_logging()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from app.llm_provider import LLMNotReadyError, get_llm  # noqa: WPS433

    model_path = os.getenv("LLM_MODEL_PATH") or os.getenv("LLM_BG1_PATH") or os.getenv("LLM_BG2_PATH")
    if model_path:
        logging.info("Configured model path: %s", model_path)
    else:
        logging.warning("LLM path variables are not configured; falling back to stub.")

    llm = get_llm()
    logging.info("Resolved LLM implementation: %s (device=%s)", llm.model_name, llm.device)

    try:
        llm.preload()
    except LLMNotReadyError as error:
        logging.error("Failed to load model: %s", error)
        return 1
    except Exception as error:  # pragma: no cover - defensive guard
        logging.exception("Unexpected error while loading model")
        return 1

    status = llm.status()
    logging.info("Model '%s' loaded on %s", status.model_name, status.device)
    if status.error:
        logging.warning("Model reported warning: %s", status.error)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
