"""Utilitaires de journalisation."""

from __future__ import annotations

import logging
from datetime import datetime


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure un logger simple avec horodatage compact."""

    logger = logging.getLogger("aerotech_etl")
    if logger.handlers:
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler()

    class Formatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            return f"[{ts}] {record.getMessage()}"

    handler.setFormatter(Formatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
