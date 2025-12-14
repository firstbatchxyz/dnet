"""Logging utilities for dnet.

Usage:
    from dnet.utils.logger import logger
    logger.info("message")

Log level and profile logging are controlled by environment variables:
    DNET_LOG=DEBUG
    DNET_PROFILE=true
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path


def _is_env_truthy(val: str | None) -> bool:
    """Return True if the string represents a truthy value (1, true, yes, on)."""
    if not val:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_log_settings() -> tuple[str, bool, str]:
    """Get log settings, trying centralized config first, falling back to env vars.

    Returns:
        Tuple of (log_level, profile_enabled, log_dir)
    """
    try:
        from dnet.config import get_settings

        settings = get_settings()
        return (
            settings.logging.log,
            settings.logging.profile,
            settings.storage.log_dir,
        )
    except Exception:
        # Fallback to raw env vars if settings can't be loaded
        # (e.g., during early init or circular import issues)
        return (
            os.getenv("DNET_LOG", "INFO"),
            _is_env_truthy(os.getenv("DNET_PROFILE", "0")),
            os.getenv("DNET_LOG_DIR", "~/.dria/dnet"),
        )


class ProfileLogFilter(logging.Filter):
    """Filter that hides [PROFILE] messages unless profiling is enabled."""

    def __init__(self, enabled: bool):
        super().__init__()
        self.enabled = enabled

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage() if hasattr(record, "getMessage") else str(record.msg)
        return not ("[PROFILE]" in msg and not self.enabled)


@lru_cache(maxsize=None)
def get_logger() -> logging.Logger:
    """Returns a configured logger for dnet.

    Log level is set by DNET_LOG env var (default INFO).
    Profile logs ([PROFILE]) are shown only if DNET_PROFILE env var is truthy.
    """
    log_level_str, profile_enabled, log_dir_str = _get_log_settings()

    log_level = getattr(logging, log_level_str.strip().upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )

    logger = logging.getLogger("dnet")
    logger.addFilter(ProfileLogFilter(profile_enabled))

    # Add file handler for crash reports
    try:
        log_dir = Path(log_dir_str).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)

        proc_name = os.path.basename(sys.argv[0])
        if "dnet-api" in proc_name:
            filename = "dnet-api.log"
        elif "dnet-shard" in proc_name:
            filename = f"dnet-shard-{os.getpid()}.log"
        else:
            filename = "dnet.log"

        log_file = log_dir / filename

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            )
        )
        logger.addHandler(file_handler)
    except Exception:
        pass

    return logger


logger = get_logger()
