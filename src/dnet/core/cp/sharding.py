"""Mode-aware sequence sharding for context parallelism.

Provides utilities for partitioning sequences across CP ranks:
- Prefill: Load-balanced 2N sharding (first+last pairs) for causal attention
- Decode: Even N-way split for uniform KV lookup compute
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx


def shard_for_mode(
    tokens_or_kv: mx.array,
    num_ranks: int,
    rank_id: int,
    mode: Literal["prefill", "decode"],
) -> tuple[mx.array, list[int]]:
    """
    Mode-aware sharding for context parallelism.

    Args:
        tokens_or_kv: Input tensor with sequence dimension at axis 0
        num_ranks: Total number of CP ranks
        rank_id: This rank's ID (0 to num_ranks-1)
        mode: "prefill" for load-balanced 2N sharding, "decode" for even splits

    Returns:
        sharded: Portion of input assigned to this rank
        indices: Original positions (for unsharding)

    Prefill sharding (2N load-balanced):
        Sequence [C0, C1, C2, C3, C4, C5, C6, C7] with 4 ranks:
        - Rank 0: [C0, C7]  (first + last)
        - Rank 1: [C1, C6]
        - Rank 2: [C2, C5]
        - Rank 3: [C3, C4]

    Decode sharding (even N-way):
        Sequence split into N equal contiguous chunks.
    """
    seq_len = tokens_or_kv.shape[0]

    if seq_len == 0:
        return tokens_or_kv, []

    if num_ranks <= 0:
        raise ValueError(f"num_ranks must be positive, got {num_ranks}")

    if not 0 <= rank_id < num_ranks:
        raise ValueError(f"rank_id {rank_id} out of range [0, {num_ranks})")

    if mode == "prefill":
        return _shard_prefill(tokens_or_kv, num_ranks, rank_id, seq_len)
    else:  # decode
        return _shard_decode(tokens_or_kv, num_ranks, rank_id, seq_len)


def _shard_prefill(
    tokens_or_kv: mx.array,
    num_ranks: int,
    rank_id: int,
    seq_len: int,
) -> tuple[mx.array, list[int]]:
    """Load-balanced 2N sharding for causal attention."""
    # Partition into 2N chunks, assign complementary pairs
    num_chunks = 2 * num_ranks
    chunk_size = seq_len // num_chunks
    remainder = seq_len % num_chunks

    # Assign chunks (i, 2N-i-1) to rank i
    chunk_a = rank_id
    chunk_b = num_chunks - rank_id - 1

    # Calculate start/end for chunk_a
    start_a = chunk_a * chunk_size + min(chunk_a, remainder)
    end_a = start_a + chunk_size + (1 if chunk_a < remainder else 0)

    # Calculate start/end for chunk_b
    start_b = chunk_b * chunk_size + min(chunk_b, remainder)
    end_b = start_b + chunk_size + (1 if chunk_b < remainder else 0)

    # Handle case where chunk_a == chunk_b (only possible when num_ranks=1)
    if chunk_a == chunk_b:
        sharded = tokens_or_kv[start_a:end_a]
        indices = list(range(start_a, end_a))
    else:
        sharded = mx.concatenate(
            [tokens_or_kv[start_a:end_a], tokens_or_kv[start_b:end_b]]
        )
        indices = list(range(start_a, end_a)) + list(range(start_b, end_b))

    return sharded, indices


def _shard_decode(
    tokens_or_kv: mx.array,
    num_ranks: int,
    rank_id: int,
    seq_len: int,
) -> tuple[mx.array, list[int]]:
    """Even N-way split for uniform decode compute."""
    chunk_size = seq_len // num_ranks
    remainder = seq_len % num_ranks

    # Distribute remainder across first 'remainder' ranks
    start = rank_id * chunk_size + min(rank_id, remainder)
    local_size = chunk_size + (1 if rank_id < remainder else 0)
    end = start + local_size

    sharded = tokens_or_kv[start:end]
    indices = list(range(start, end))

    return sharded, indices


def unshard(
    sharded_chunks: list[mx.array],
    indices_per_rank: list[list[int]],
    total_seq_len: int,
) -> mx.array:
    """
    Reconstruct full sequence from sharded chunks.

    Args:
        sharded_chunks: List of sharded tensors, one per rank
        indices_per_rank: List of index lists from shard_for_mode
        total_seq_len: Total sequence length

    Returns:
        Reconstructed tensor with original ordering
    """
    if not sharded_chunks:
        raise ValueError("sharded_chunks cannot be empty")

    # Get shape info from first chunk
    sample = sharded_chunks[0]
    rest_shape = sample.shape[1:]
    dtype = sample.dtype

    # Create output buffer
    output = mx.zeros((total_seq_len,) + rest_shape, dtype=dtype)

    # Scatter chunks back to original positions
    # Note: Using .add() even though indices are disjoint because MLX ArrayAt
    # doesn't have .set() method. Since indices don't overlap, this is equivalent.
    for chunk, indices in zip(sharded_chunks, indices_per_rank):
        if len(indices) != chunk.shape[0]:
            raise ValueError(
                f"Chunk size {chunk.shape[0]} != indices length {len(indices)}"
            )
        for i, idx in enumerate(indices):
            output = output.at[idx].add(chunk[i])

    return output
