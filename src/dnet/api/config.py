"""API configuration (deprecated: use dnet.config instead).

This module provides backward compatibility with the old dataclass-based config.
New code should import from dnet.config directly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

# Emit deprecation warning when this module is imported
warnings.warn(
    "dnet.api.config is deprecated, use dnet.config instead",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class ApiConfig:
    """API server configuration (deprecated)."""

    http_port: int
    grpc_port: int
    compression_pct: float = 0.0

    @classmethod
    def from_settings(cls) -> "ApiConfig":
        """Create from centralized settings."""
        from dnet.config import get_settings

        s = get_settings().api
        return cls(s.http_port, s.grpc_port, s.compression_pct)


@dataclass
class ClusterConfig:
    """Cluster configuration (deprecated)."""

    discovery_port: int = 0  # 0 means dynamic/default

    @classmethod
    def from_settings(cls) -> "ClusterConfig":
        """Create from centralized settings."""
        from dnet.config import get_settings

        s = get_settings().api
        return cls(s.discovery_port)


@dataclass
class InferenceConfig:
    """Inference configuration (deprecated)."""

    max_concurrent_requests: int = 100

    @classmethod
    def from_settings(cls) -> "InferenceConfig":
        """Create from centralized settings."""
        from dnet.config import get_settings

        s = get_settings().api
        return cls(s.max_concurrent_requests)
