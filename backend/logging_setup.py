from __future__ import annotations

import json
import logging
import sys
import time

_REDACTED_KEYS = {"api_key", "audio", "audio_data", "raw_audio", "content"}


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.Logger) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }
        extra = getattr(record, "fields", None)
        if extra:
            for key, value in extra.items():
                if key in _REDACTED_KEYS:
                    continue
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("voicepilot")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: int, event: str, **fields: object) -> None:
    logger.log(level, event, extra={"fields": fields})
