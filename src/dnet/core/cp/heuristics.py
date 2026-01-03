"""Algorithm selection heuristics for context parallelism.

Provides a greedy heuristic for selecting the optimal CP algorithm based on:
- Context length and cache hit rate
- Batch size
- Number of query/KV heads (GQA ratio)
- Number of CP ranks

This is a v1 hardcoded heuristic. Future versions will use a solver-based
approach for more accurate predictions.
"""

from __future__ import annotations

from enum import StrEnum


class CPAlgorithm(StrEnum):
    """Context parallelism algorithm selection."""

    SINGLE_DEVICE = "single_device"  # No CP, run on single device
    PASS_KV = "pass_kv"  # Rotate KV blocks (best for prefill)
    PASS_Q = "pass_q"  # Rotate Q blocks with All2All
    RING_REDUCE = "ring_reduce"  # Rotate Q with ring reduction (best for decode)


def select_algorithm(
    new_tokens: int,
    cached_tokens: int,
    batch_size: int,
    num_ranks: int,
    num_q_heads: int,
    num_kv_heads: int,
    context_parallel_enabled: bool,
    min_context_for_cp: int = 32768,
    min_tokens_for_pass_kv: int = 256,
    gqa_threshold: float | None = None,
) -> CPAlgorithm:
    """
    Greedy heuristic for selecting CP algorithm.

    Decision tree:
    1. Skip CP for small contexts or if disabled
    2. Decode mode (T <= batch_size) → ring_reduce (avoid All2All)
    3. Prefill with high cache hit → pass_q (Q smaller than KV)
    4. Full prefill → pass_kv (enough compute to hide comm)

    Args:
        new_tokens: Number of new tokens to process (T)
        cached_tokens: Number of tokens already in KV cache (P)
        batch_size: Current batch size
        num_ranks: Number of CP ranks
        num_q_heads: Number of query heads
        num_kv_heads: Number of KV heads (for GQA models)
        context_parallel_enabled: Whether CP is enabled in config
        min_context_for_cp: Minimum context to use CP (default 32K)
        min_tokens_for_pass_kv: Minimum new tokens for pass-KV (default 256)
        gqa_threshold: Cache miss rate threshold (default: 2 * NKV / NH)

    Returns:
        Selected algorithm from CPAlgorithm enum
    """
    total_context = new_tokens + cached_tokens

    # Rule 1: Skip CP for small contexts or if disabled
    if not context_parallel_enabled or total_context < min_context_for_cp:
        return CPAlgorithm.SINGLE_DEVICE

    # Rule 2: Single rank is always single device
    if num_ranks <= 1:
        return CPAlgorithm.SINGLE_DEVICE

    # Rule 3: Decode mode (T=1 per sequence in batch typically)
    # Heuristic: if new_tokens <= batch_size, likely decode
    if new_tokens <= batch_size:
        return CPAlgorithm.RING_REDUCE  # Avoid All2All for decode

    # Calculate cache miss rate
    miss_rate = new_tokens / total_context if total_context > 0 else 1.0

    # Compute GQA threshold if not provided
    # Threshold from paper: 2 * NKV / NH (e.g., 2*8/128 = 0.125 for Llama)
    if gqa_threshold is None:
        if num_q_heads > 0:
            gqa_threshold = 2.0 * num_kv_heads / num_q_heads
        else:
            gqa_threshold = 0.125  # Default fallback

    # Rule 4: Prefill with high cache hit (partial prefill)
    # When miss rate is low, Q is much smaller than full KV
    if miss_rate < gqa_threshold:
        return CPAlgorithm.PASS_Q

    # Rule 5: Full prefill or sufficient new tokens
    # pass-KV has enough compute to hide KV communication
    if new_tokens >= min_tokens_for_pass_kv:
        return CPAlgorithm.PASS_KV

    # Fallback for edge cases (short prefill with low cache hit)
    return CPAlgorithm.PASS_Q


def estimate_algorithm_latency(
    algorithm: CPAlgorithm,
    new_tokens: int,
    cached_tokens: int,
    num_ranks: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    flops_per_sec: float,
    bandwidth_bytes_per_sec: float,
) -> float:
    """
    Estimate latency for a given algorithm (for solver integration).

    This is a simplified model for v1. Actual latency depends on:
    - Overlap between compute and communication
    - Memory bandwidth
    - Kernel efficiency

    Args:
        algorithm: Selected algorithm
        new_tokens: Number of new tokens
        cached_tokens: Number of cached tokens
        num_ranks: Number of CP ranks
        num_q_heads: Query heads
        num_kv_heads: KV heads
        head_dim: Dimension per head
        flops_per_sec: Device compute throughput
        bandwidth_bytes_per_sec: Inter-device bandwidth

    Returns:
        Estimated latency in seconds
    """
    total_context = new_tokens + cached_tokens
    bytes_per_element = 2  # bfloat16

    if algorithm == CPAlgorithm.SINGLE_DEVICE:
        # Full attention compute
        attn_flops = 2 * new_tokens * total_context * num_q_heads * head_dim
        return attn_flops / flops_per_sec

    tokens_per_rank = total_context // num_ranks

    if algorithm == CPAlgorithm.PASS_KV:
        # Compute: distributed across ranks
        attn_flops = 2 * new_tokens * total_context * num_q_heads * head_dim
        compute_time = attn_flops / (flops_per_sec * num_ranks)

        # Communication: KV blocks rotated N-1 times
        kv_size = tokens_per_rank * num_kv_heads * head_dim * bytes_per_element * 2
        comm_time = (num_ranks - 1) * kv_size / bandwidth_bytes_per_sec

        # Overlap: max of compute and comm (simplified)
        return max(compute_time, comm_time)

    elif algorithm == CPAlgorithm.PASS_Q:
        # Compute: same as pass-KV
        attn_flops = 2 * new_tokens * total_context * num_q_heads * head_dim
        compute_time = attn_flops / (flops_per_sec * num_ranks)

        # Communication: Q blocks + All2All
        q_size = (new_tokens // num_ranks) * num_q_heads * head_dim * bytes_per_element
        ring_comm = (num_ranks - 1) * q_size / bandwidth_bytes_per_sec

        # All2All: O(N^2) communication pattern
        output_size = new_tokens * num_q_heads * head_dim * bytes_per_element
        all2all_time = output_size / bandwidth_bytes_per_sec  # Simplified

        return max(compute_time, ring_comm) + all2all_time

    else:  # RING_REDUCE
        # Compute: same as others
        attn_flops = 2 * new_tokens * total_context * num_q_heads * head_dim
        compute_time = attn_flops / (flops_per_sec * num_ranks)

        # Communication: partial outputs + merge stats
        # Each step passes output + max_score + log_sum_exp
        output_per_rank = (new_tokens // num_ranks) * num_q_heads * head_dim
        stats_per_rank = (new_tokens // num_ranks) * num_q_heads * 2  # max + lse
        bytes_per_step = (output_per_rank + stats_per_rank) * bytes_per_element
        ring_time = (num_ranks - 1) * bytes_per_step / bandwidth_bytes_per_sec

        return max(compute_time, ring_time)
