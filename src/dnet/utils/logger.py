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
    logLevelEnv = os.getenv("DNET_LOG", "INFO").strip().upper()
    logLevel = logging.INFO  # default
    if logLevelEnv in logging._nameToLevel:
        logLevel = logging._nameToLevel[logLevelEnv]

    logging.basicConfig(
        level=logLevel,
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

    # Add file handler for crash reports
    try:
        from pathlib import Path
        import sys

        log_dir = Path.home() / ".dria" / "dnet"
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
        file_handler.setLevel(logLevel)
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
