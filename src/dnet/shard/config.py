from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class KVCacheConfig:
    """Quantized/unquantized KV cache configuration.

    - mode: "fp16" (default), "8bit", "4bit", or "quant" (custom bits)
    - bits: number of bits when mode == "quant" (1..8)
    - group_size: grouping for quantized cache (e.g., 64)
    """

    mode: str = "8bit"
    bits: int = 8
    group_size: int = 64
    kv_ttl_s: float = 30.0


@dataclass
class ComputeConfig:
    """Compute configuration for a shard process."""

    # We shouldn't need prefetch_mode here anymore
    prefetch_mode = "off"
    mxload_fastpath: bool = True
    input_pool_mb: int = 512
    output_pool_mb: int = 512
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)


@dataclass
class TransportConfig:
    """Transport configuration for a shard process."""

    wire_mode: str = "q8_dense"
    wire_dtype: str = "fp16"
    streaming: bool = True
    stream_backoff_s: float = 0.5
    stream_idle_s: float = 2.0
    send_retries: int = 3
    explicit_eor: bool = False
    compress: bool = False
    compress_min_bytes: int = 65536


@dataclass
class TopologyConfig:
    """Topology configuration for a shard process."""

    resident_windows: int = 1
    warmup_windows: int = 1
    x_stats: bool = False
