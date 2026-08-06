"""
utils.py
--------
Shared helpers: logging setup and environment/config loading.

Why this file exists:
Every module in this project needs to log what it's doing and needs access
to secrets like the FRED API key WITHOUT hardcoding them into the code.
Centralizing both here means every other module just does:
    from src.utils import get_logger, load_config
    
(so when something breaks at 11pm, you can see WHERE it broke)
"""

import logging
import os
import sys
from dotenv import load_dotenv


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger. Call this at the top of every module:
        logger = get_logger(__name__)
    __name__ automatically becomes e.g. 'src.data_loader', so log lines
    tell you exactly which file produced them.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers on re-import
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def load_config() -> dict:
    """
    Loads environment variables from a local .env file (never committed
    to git — see .gitignore) and returns them as a dict.

    Create a file named `.env` in the project root containing:
        FRED_API_KEY=your_key_here
    """
    load_dotenv()

    config = {
        "FRED_API_KEY": os.getenv("FRED_API_KEY"),
    }

    return config
