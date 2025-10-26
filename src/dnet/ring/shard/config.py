"""Central configuration for shard runtime.

Minimal, mode-driven configuration to replace scattered os.getenv() reads.
Keep defaults simple; allow future override without touching call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KVCacheConfig:
    """Quantized/unquantized KV cache configuration.

    - mode: "fp16" (default), "int8", "int4", or "quant" (custom bits)
    - bits: number of bits when mode == "quant" (1..8)
    - group_size: grouping for quantized cache (e.g., 64)
    """

    mode: str = "fp16"
    bits: int = 8
    group_size: int = 64
    kv_ttl_s: float = 30.0


@dataclass
class ShardConfig:
    """Configuration for a shard process.

    Notes:
    - Mode selects sensible presets. Use "fit" when model fits in RAM;
      use "offload" when relying on on-demand weight IO.
    - Values here are the effective knobs the shard code reads; some modules
      outside shard still consult env vars — we mirror these values into env
      in RingShardNode for backward compatibility.
    """

    # Operation mode
    mode: str = "fit"  # "fit" | "offload"

    resident_windows: int = 2
    lazy_params: bool = False


    wire_dtype: str = "fp16"

    # Warmup
    warmup_windows: int = 1

    # Streaming and TX behavior
    streaming: bool = False
    stream_backoff_s: float = 0.5
    stream_idle_s: float = 2.0
    send_retries: int = 3
    explicit_eor: bool = False

    # Wire compression
    compress: bool = False
    compress_min_bytes: int = 65536

    # Debug/diagnostics
    x_stats: bool = False

    # KV cache quantization
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)

    # Weight loading behavior
    # Enable mx.load fast-path only for explicitly repacked files/windows.
    # Default False to avoid loading full multi-layer shard files in offload mode.
    mxload_fastpath: bool = False
    # IO strategy: True for sequential (no background prefetch/overlap)
    sequential_io: bool = False

    # Activation pool sizes (MB)
    input_pool_mb: int = 512
    output_pool_mb: int = 512

    @staticmethod
    def for_mode(mode: str) -> "ShardConfig":
        m = (mode or "fit").strip().lower()
        if m == "offload":
            # Focus on minimal RAM usage; allow on-demand reads
            return ShardConfig(
                mode="offload",
                resident_windows=1,
                lazy_params=True,
                wire_dtype="fp16",
                warmup_windows=1,
                streaming=False,
                compress=False,
                mxload_fastpath=True,  # Use mx.load fast-path with repacked per-layer/per-window files
                sequential_io=True,
                input_pool_mb=256,
                output_pool_mb=256,
            )
        # Default: fit-in-memory preset
        return ShardConfig(
            mode="fit",
            resident_windows=9999,
            lazy_params=False,
            wire_dtype="fp16",
            warmup_windows=1,
            streaming=True,
            compress=False,
            mxload_fastpath=False,
            sequential_io=False,
            input_pool_mb=512,
            output_pool_mb=512,
        )
