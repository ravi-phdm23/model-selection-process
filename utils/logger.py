"""Logging utilities for the pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        """Format log record with colors."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname:8s}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logger(
    log_file: Optional[Path] = None,
    level: str = 'INFO',
    console_output: bool = True,
    name: str = 'pipeline'
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        log_file: Path to log file (None = no file logging)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        console_output: Whether to output to console
        name: Logger name

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = ColoredFormatter(
        '%(levelname)s %(message)s'
    )

    # File handler (detailed format)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Console handler (simple format with colors)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = 'pipeline') -> logging.Logger:
    """
    Get existing logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
