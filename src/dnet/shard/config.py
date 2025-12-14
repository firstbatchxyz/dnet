"""Shard configuration (deprecated: use dnet.config instead).

This module provides backward compatibility with the old dataclass-based config.
New code should import from dnet.config directly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

# Emit deprecation warning when this module is imported
warnings.warn(
    "dnet.shard.config is deprecated, use dnet.config instead",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class KVCacheConfig:
    """Quantized/unquantized KV cache configuration (deprecated).

    - mode: "fp16" (default), "8bit", "4bit", or "quant" (custom bits)
    - bits: number of bits when mode == "quant" (1..8)
    - group_size: grouping for quantized cache (e.g., 64)
    """

    mode: str = "8bit"
    bits: int = 8
    group_size: int = 64
    kv_ttl_s: float = 30.0

    @classmethod
    def from_settings(cls) -> "KVCacheConfig":
        """Create from centralized settings."""
        from dnet.config import get_settings

        s = get_settings().kv_cache
        return cls(s.mode, s.bits, s.group_size, s.ttl_s)


@dataclass
class ComputeConfig:
    """Compute configuration for a shard process (deprecated)."""

    prefetch_mode: str = "off"
    mxload_fastpath: bool = True
    input_pool_mb: int = 512
    output_pool_mb: int = 512
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)

    @classmethod
    def from_settings(cls) -> "ComputeConfig":
        """Create from centralized settings."""
        from dnet.config import get_settings

        s = get_settings().compute
        return cls(
            prefetch_mode="off",
            mxload_fastpath=s.mxload_fastpath,
            input_pool_mb=s.input_pool_mb,
            output_pool_mb=s.output_pool_mb,
            kv_cache=KVCacheConfig.from_settings(),
        )


@dataclass
class TransportConfig:
    """Transport configuration for a shard process (deprecated)."""

    wire_dtype: str = "fp16"
    streaming: bool = True
    stream_backoff_s: float = 0.5
    stream_idle_s: float = 2.0
    send_retries: int = 3
    explicit_eor: bool = False
    compress: bool = False
    compress_min_bytes: int = 65536

    @classmethod
    def from_settings(cls) -> "TransportConfig":
        """Create from centralized settings."""
        from dnet.config import get_settings

        s = get_settings().transport
        return cls(
            wire_dtype=s.wire_dtype,
            streaming=s.streaming,
            stream_backoff_s=s.stream_backoff_s,
            stream_idle_s=s.stream_idle_s,
            send_retries=s.send_retries,
            explicit_eor=False,
            compress=s.compress,
            compress_min_bytes=s.compress_min_bytes,
        )


@dataclass
class TopologyConfig:
    """Topology configuration for a shard process (deprecated)."""

    resident_windows: int = 1
    warmup_windows: int = 1
    x_stats: bool = False

    @classmethod
    def from_settings(cls) -> "TopologyConfig":
        """Create from centralized settings."""
        from dnet.config import get_settings

        s = get_settings().topology
        return cls(
            resident_windows=s.resident_windows,
            warmup_windows=s.warmup_windows,
            x_stats=s.x_stats,
        )
