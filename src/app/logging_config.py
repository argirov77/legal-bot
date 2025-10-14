import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

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

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._RESERVED_KEYS and not key.startswith("_")
        }
        if extra:
            payload.update(extra)
        if "audit_filename" in payload and "filename" not in payload:
            payload["filename"] = payload.pop("audit_filename")

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application logging with a JSON formatter."""

    root_logger = logging.getLogger()
    if getattr(root_logger, "_app_configured", False):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())

    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger._app_configured = True  # type: ignore[attr-defined]


_AUDIT_LOG_PATH = Path("logs") / "ingest_audit.log"


def get_ingest_audit_logger() -> logging.Logger:
    """Return a logger configured to write ingest audit events."""

    logger = logging.getLogger("app.ingest.audit")
    if getattr(logger, "_audit_configured", False):
        return logger

    _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(_AUDIT_LOG_PATH, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(JsonFormatter())

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    logger._audit_configured = True  # type: ignore[attr-defined]
    return logger
