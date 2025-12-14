"""gRPC configuration for dnet.

Uses centralized settings from dnet.config for configuration values.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dnet.config import GrpcSettings


def _get_grpc_settings() -> "GrpcSettings":
    """Get gRPC settings, falling back to defaults if config can't be loaded."""
    try:
        from dnet.config import get_settings

        return get_settings().grpc
    except Exception:
        # Fallback to defaults if settings can't be loaded
        from dnet.config import GrpcSettings

        return GrpcSettings()


@lru_cache
def get_grpc_options() -> list[tuple[str, int]]:
    """Build gRPC channel options from settings."""
    s = _get_grpc_settings()
    return [
        ("grpc.max_send_message_length", s.max_message_length),
        ("grpc.max_receive_message_length", s.max_message_length),
        # Concurrency for streaming
        ("grpc.max_concurrent_streams", s.max_concurrent_streams),
        # Conservative keepalive/pings to avoid GOAWAY too_many_pings
        ("grpc.keepalive_time_ms", s.keepalive_time_ms),
        ("grpc.keepalive_timeout_ms", s.keepalive_timeout_ms),
        ("grpc.keepalive_permit_without_calls", 0),
        ("grpc.http2.min_time_between_pings_ms", 120000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.bdp_probe", 0),  # disable BDP probe to reduce pinging
        # Avoid any interference from HTTP proxies for direct ring links
        ("grpc.enable_http_proxy", 0),
    ]


# Backward compatibility: module-level constant
# Note: This is evaluated at import time, so changes to env vars after import
# won't be reflected unless get_grpc_options() is called directly.
GRPC_MAX_MESSAGE_LENGTH = _get_grpc_settings().max_message_length
GRPC_AIO_OPTIONS = get_grpc_options()
