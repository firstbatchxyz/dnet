from __future__ import annotations

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED_FLAG = "_dnet_api_logger_configured"
logging.getLogger("urllib3").setLevel(logging.WARNING)

def get_api_logger() -> logging.Logger:
    log = logging.getLogger("dnet.api")
    if getattr(log, _CONFIGURED_FLAG, False):
        return log

    # Level from env, fallback INFO
    level_name = (os.getenv("DNET_API_LOG", "INFO") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    #log.setLevel(level)
    log.setLevel(logging.DEBUG)

    # Do not bubble to root (console)
    log.propagate = False

    # Ensure logs directory
    try:
        Path("logs").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Attach a rotating file handler
    try:
        fh = RotatingFileHandler("logs/api.log", maxBytes=10000000, backupCount=5)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"
        )
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except Exception:
        # As a last resort, attach a NullHandler to avoid 'No handler' warnings
        log.addHandler(logging.NullHandler())

    setattr(log, _CONFIGURED_FLAG, True)
    return log

