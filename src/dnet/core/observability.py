"""Observability and profiling utilities for dnet.

Uses centralized settings from dnet.config for configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..utils.logger import logger


@dataclass(frozen=True)
class ObsSettings:
    """Observability settings (immutable)."""

    enabled: bool
    sync_per_layer: bool
    sync_every_n: int


def _truthy(val: str | None) -> bool:
    """Return True for truthy string values."""
    if not val:
        return False
    s = val.strip().lower()
    return s in {"1", "true", "yes", "on"}


def load_settings() -> ObsSettings:
    """Load observability settings from centralized config.

    Profile can be enabled via:
    - DNET_OBS_ENABLED=true
    - DNET_PROFILE=true (legacy, also enables profile logging)
    - RING_PROFILE=1, PROFILE=1, RUN_PROFILE=1, SHARD_PROFILE=1 (legacy compat)
    """
    try:
        from dnet.config import get_settings

        settings = get_settings()
        obs = settings.observability
        log_settings = settings.logging

        # Profile enabled if centralized setting or legacy env vars are set
        enabled = (
            obs.enabled
            or log_settings.profile
            or any(
                _truthy(os.getenv(k))
                for k in ("RING_PROFILE", "PROFILE", "RUN_PROFILE", "SHARD_PROFILE")
            )
        )

        # Per-layer sync: default to enabled when profiling, else use setting
        sync_per_layer = obs.sync_per_layer if obs.sync_per_layer else enabled

        return ObsSettings(
            enabled=enabled,
            sync_per_layer=sync_per_layer,
            sync_every_n=max(0, obs.sync_every_n),
        )
    except Exception:
        # Fallback to legacy behavior if settings can't be loaded
        enabled = any(
            _truthy(os.getenv(k))
            for k in ("RING_PROFILE", "PROFILE", "RUN_PROFILE", "SHARD_PROFILE")
        )

        spe = os.getenv("RING_SYNC_PER_LAYER")
        sync_per_layer = enabled if spe is None else _truthy(spe)

        try:
            sync_every_n = int((os.getenv("RING_SYNC_EVERY_N") or "0").strip())
        except Exception:
            sync_every_n = 0

        return ObsSettings(
            enabled=enabled,
            sync_per_layer=sync_per_layer,
            sync_every_n=max(0, sync_every_n),
        )


class Profiler:
    """Conditional profiler that logs only when enabled."""

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
    """Create a Profiler instance."""
    return Profiler(enabled)


__all__ = ["ObsSettings", "load_settings", "Profiler", "make_profiler"]
