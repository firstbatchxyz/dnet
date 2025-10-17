"""Central configuration for shard runtime.

Minimal, mode-driven configuration to replace scattered os.getenv() reads.
Keep defaults simple; allow future override without touching call sites.
"""

from __future__ import annotations

from dataclasses import dataclass


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

    # Memory and file-cache behavior
    resident_windows: int = 2  # windows to keep resident in memory
    file_cache_mode: str = "auto"  # "auto" | "none"
    file_dict_cap: int | None = None  # explicit capacity for file-cache dict
    materialize_prefetch: bool = False  # materialize weights into RAM during prefetch
    lazy_params: bool = False  # lazily allocate module params (model-specific)

    # Prefetch policy
    prefetch_mode: str = "off"  # off | sequential | full
    prefetch_touch: str = "none"  # none | sum | stripe
    prefetch_async: bool = True
    prefetch_fraction: float = 0.25  # fraction for stripe mode
    prefetch_budget_ms: float = 25.0  # guardrail on touch work per layer

    # IO & wire
    file_io_direct: bool = False  # mmapped direct IO (bypass caching)
    wire_dtype: str = "fp16"  # fp16 | bf16
    kv_ttl_s: float = 30.0  # KV store TTL for cross-node comms

    # Warmup
    warmup_windows: int = 1

    # Streaming and TX behavior
    stream_backoff_s: float = 0.5
    stream_idle_s: float = 2.0
    send_retries: int = 3
    explicit_eor: bool = False

    # Debug/diagnostics
    x_stats: bool = False

    @staticmethod
    def for_mode(mode: str) -> "ShardConfig":
        m = (mode or "fit").strip().lower()
        if m == "offload":
            # Focus on minimal RAM usage; allow on-demand reads
            return ShardConfig(
                mode="offload",
                resident_windows=1,
                file_cache_mode="auto",
                file_dict_cap=1,
                materialize_prefetch=False,
                lazy_params=True,
                prefetch_mode="off",
                prefetch_touch="none",
                prefetch_async=True,
                prefetch_fraction=0.25,
                prefetch_budget_ms=25.0,
                file_io_direct=False,
                wire_dtype="fp16",
                kv_ttl_s=30.0,
                warmup_windows=1,
            )
        # Default: fit-in-memory preset
        return ShardConfig(
            mode="fit",
            resident_windows=9999,
            file_cache_mode="none",
            file_dict_cap=None,
            materialize_prefetch=False,
            lazy_params=False,
            prefetch_mode="off",
            prefetch_touch="none",
            prefetch_async=True,
            prefetch_fraction=0.25,
            prefetch_budget_ms=25.0,
            file_io_direct=False,
            wire_dtype="fp16",
            kv_ttl_s=30.0,
            warmup_windows=1,
        )
