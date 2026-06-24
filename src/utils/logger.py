"""Logging utilities for the adaptive mapping framework."""
import logging
import sys
from pathlib import Path


def setup_logger(name, log_dir="./logs", level=logging.INFO):
    """Setup logger with file and console handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
