"""Logging utilities for dnet."""

import logging
import os
from functools import lru_cache


@lru_cache(maxsize=None)
def get_logger() -> logging.Logger:
    """Get configured logger for dnet.

    Profile logs (containing [PROFILE]) are only shown when PROFILE env var is set.

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )

    class _ProfileLogFilter(logging.Filter):
        """Filter that controls profile log visibility."""

        def __init__(self, enabled: bool) -> None:
            super().__init__()
            self.enabled = enabled

        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                msg = str(record.msg)
            if "[PROFILE]" in msg and not self.enabled:
                return False
            return True

    # PROFILE env: enable profile logs when set to a truthy value
    prof_env = os.getenv("PROFILE", "0").strip().lower()
    profile_enabled = prof_env in {"1", "true", "yes", "on"}

    logger = logging.getLogger("dnet")
    logger.addFilter(_ProfileLogFilter(profile_enabled))
    return logger


logger = get_logger()
