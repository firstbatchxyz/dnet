from __future__ import annotations

import os
from dataclasses import dataclass
from ..utils.logger import logger


@dataclass(frozen=True)
class ObsSettings:
    enabled: bool
    sync_per_layer: bool
    sync_every_n: int


def _truthy(val: str | None) -> bool:
    if not val:
        return False
    s = val.strip().lower()
    return s in {"1", "true", "yes", "on"}


def load_settings() -> ObsSettings:
    # Profile enable flags (union of common envs)
    enabled = any(
        _truthy(os.getenv(k))
        for k in ("RING_PROFILE", "PROFILE", "RUN_PROFILE", "SHARD_PROFILE")
    )

    # Per-layer sync: default to enabled when profiling, else off unless explicitly set
    spe = os.getenv("RING_SYNC_PER_LAYER")
    sync_per_layer = enabled if spe is None else _truthy(spe)

    # Sync cadence inside a window (0 disables)
    try:
        sync_every_n = int((os.getenv("RING_SYNC_EVERY_N") or "0").strip())
    except Exception:
        sync_every_n = 0
    sync_every_n = max(0, sync_every_n)

    return ObsSettings(
        enabled=enabled,
        sync_per_layer=sync_per_layer,
        sync_every_n=sync_every_n,
    )


__all__ = ["ObsSettings", "load_settings"]


class Profiler:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def info(self, msg: str, *args, **kwargs) -> None:
        if self.enabled:
            logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        if self.enabled:
            logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        if self.enabled:
            logger.warning(msg, *args, **kwargs)


def make_profiler(enabled: bool) -> Profiler:
    return Profiler(enabled)


__all__.extend(["Profiler", "make_profiler"])
