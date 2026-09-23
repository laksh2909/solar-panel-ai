"""
Logging utility for Solar Panel Fault Detection.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "solar_ai", log_file: Path | str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up and configures a standardized logger.
    
    Args:
        name: Name of the logger.
        log_file: Optional path to output log file.
        level: Logging level (default INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
