"""
Logging utilities for dnet.

Usage:
    from dnet.utils.logger import logger
    logger.info("message")
    # Log level and profile logging are controlled by .env or environment variables:
    # DNET_LOG=DEBUG
    # PROFILE=1
"""

import logging
import os
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def is_env_truthy(val: str) -> bool:
    """Return True if the string represents a truthy value (1, true, yes, on)."""
    return val.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=None)
def get_logger() -> logging.Logger:
    """
    Returns a configured logger for dnet.
    Log level is set by DNET_LOG env var (default INFO).
    Profile logs ([PROFILE]) are shown only if PROFILE env var is truthy.
    """
    log_level_str = os.getenv("DNET_LOG", "INFO").strip().upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )

    class ProfileLogFilter(logging.Filter):
        def __init__(self, enabled: bool):
            super().__init__()
            self.enabled = enabled

        def filter(self, record: logging.LogRecord) -> bool:
            msg = (
                record.getMessage()
                if hasattr(record, "getMessage")
                else str(record.msg)
            )
            return not ("[PROFILE]" in msg and not self.enabled)

    profile_enabled = is_env_truthy(os.getenv("PROFILE", "0"))
    logger = logging.getLogger("dnet")
    logger.addFilter(ProfileLogFilter(profile_enabled))
    return logger


logger = get_logger()
