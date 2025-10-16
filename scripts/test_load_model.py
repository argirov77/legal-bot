#!/usr/bin/env python3
"""Helper script that exercises the LLM loading path in isolation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
src_str = str(SRC_ROOT)
if src_str not in sys.path:
    sys.path.insert(0, src_str)

from app.logging_config import configure_logging
from app.llm_provider import get_llm_status, load_llm_on_startup


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force model loading attempt even if INSTALL_HEAVY is disabled.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    if args.force:
        os.environ.setdefault("FORCE_LOAD_ON_START", "true")

    configure_logging()
    attempted, error = load_llm_on_startup()
    status = get_llm_status()

    print(json.dumps(asdict(status), ensure_ascii=False, indent=2))

    if attempted and error is not None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
