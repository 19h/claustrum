"""Logging configuration for CLAUSTRUM.

Provides structured logging with rich formatting for development
and JSON output for production deployments.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Any

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_output: bool = False,
    rich_traceback: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_output: If True, output logs in JSON format
        rich_traceback: If True, use rich formatting for tracebacks
    """
    # Remove default handler
    logger.remove()

    # Console format
    if json_output:
        console_format = "{message}"
        logger.add(
            sys.stderr,
            format=console_format,
            level=level,
            serialize=True,
        )
    else:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stderr,
            format=console_format,
            level=level,
            colorize=True,
            backtrace=rich_traceback,
            diagnose=rich_traceback,
        )

    # File output
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="100 MB",
            retention="7 days",
            compression="gz",
        )


def get_logger(name: str) -> Any:
    """Get a logger instance with the given name.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        Logger instance configured for the module
    """
    return logger.bind(name=name)


class LogContext:
    """Context manager for adding contextual information to logs.

    Example:
        with LogContext(binary_hash="abc123", function_addr=0x1000):
            logger.info("Processing function")
    """

    def __init__(self, **kwargs: Any):
        self.context = kwargs
        self._token: Optional[int] = None

    def __enter__(self) -> "LogContext":
        self._token = logger.configure(extra=self.context)  # type: ignore[assignment]
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            logger.configure(extra={})


# Progress logging helpers
def log_progress(
    current: int,
    total: int,
    description: str = "",
    interval: int = 100,
) -> None:
    """Log progress at specified intervals.

    Args:
        current: Current item number
        total: Total number of items
        description: Optional description
        interval: Log every N items
    """
    if current % interval == 0 or current == total:
        percentage = (current / total) * 100
        logger.info(f"{description}: {current}/{total} ({percentage:.1f}%)")


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    """Log a dictionary of metrics.

    Args:
        metrics: Dictionary of metric names to values
        prefix: Optional prefix for the log message
    """
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    message = " | ".join(parts)
    if prefix:
        message = f"{prefix}: {message}"
    logger.info(message)
