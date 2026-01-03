"""Context Parallelism core utilities.

This package provides the core building blocks for context parallelism:
- sharding: Mode-aware sequence partitioning (prefill vs decode)
- merge_attention: Numerically stable merging of partial attention outputs
- heuristics: Algorithm selection (pass-KV, pass-Q, ring-reduce)
- ring_comm: Ring communication primitives

Note: sharding and merge_attention require MLX (macOS only).
      heuristics works on all platforms.
"""

# Platform-independent imports (always available)
from dnet.core.cp.heuristics import select_algorithm, CPAlgorithm
from dnet.core.cp.ring_comm import (
    CPRingCommunicator,
    RingNeighbors,
    CPRingServiceServicer,
    start_cp_ring_server,
)


# MLX-dependent imports (only available on macOS)
# These are lazy-imported to allow heuristics to work on other platforms
def __getattr__(name: str):
    """Lazy import for MLX-dependent modules."""
    if name in ("shard_for_mode", "unshard"):
        from dnet.core.cp.sharding import shard_for_mode, unshard

        return shard_for_mode if name == "shard_for_mode" else unshard
    elif name in (
        "PartialAttentionOutput",
        "merge_partial_attention",
        "merge_two_partials",
    ):
        from dnet.core.cp.merge_attention import (
            PartialAttentionOutput,
            merge_partial_attention,
            merge_two_partials,
        )

        if name == "PartialAttentionOutput":
            return PartialAttentionOutput
        elif name == "merge_partial_attention":
            return merge_partial_attention
        else:
            return merge_two_partials
    raise AttributeError(f"module 'dnet.core.cp' has no attribute {name!r}")


__all__ = [
    "shard_for_mode",
    "unshard",
    "PartialAttentionOutput",
    "merge_partial_attention",
    "merge_two_partials",
    "select_algorithm",
    "CPAlgorithm",
    "CPRingCommunicator",
    "RingNeighbors",
    "CPRingServiceServicer",
    "start_cp_ring_server",
]
