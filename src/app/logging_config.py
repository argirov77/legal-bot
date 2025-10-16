"""Application logging configuration utilities."""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MinimalJSONFormatter(logging.Formatter):
    """Serialize log records to a compact JSON string."""

    _RESERVED_KEYS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - inherited
        timestamp = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

        log_record: dict[str, Any] = {
            "ts": timestamp,
            "timestamp": timestamp,
            "level": record.levelname,
            "module": record.name,
        }

        message = record.getMessage()
        if isinstance(record.msg, dict):
            log_record.update(record.msg)
        elif message:
            log_record["message"] = message

        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in self._RESERVED_KEYS or key.startswith("_"):
                continue
            log_record[key] = value

        return json.dumps(log_record, ensure_ascii=False)


def configure_logging() -> None:
    """Configure application logging with JSON formatting."""

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": MinimalJSONFormatter}},
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                },
                "ingest_audit": {
                    "class": "logging.FileHandler",
                    "filename": str(log_dir / "ingest_audit.log"),
                    "mode": "a",
                    "encoding": "utf-8",
                    "formatter": "json",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
            "loggers": {
                "app.ingest.audit": {
                    "level": "INFO",
                    "handlers": ["ingest_audit"],
                    "propagate": False,
                }
            },
        }
    )
