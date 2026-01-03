"""
CP KV Synchronization: AllGather for KV cache across ranks.

After each layer's forward pass, each rank has KV for its local chunk only.
This module provides sync_kv_cache() to AllGather KV from all ranks,
so each rank can attend to the full sequence.

The sync is called after each layer, enabling full context attention.
"""

from __future__ import annotations

import asyncio
from typing import Optional, TYPE_CHECKING

import mlx.core as mx
import numpy as np

from dnet.utils.logger import logger

if TYPE_CHECKING:
    from dnet.core.cp.ring_comm import CPRingCommunicator


def serialize_kv_layer(kv_cache_layer) -> bytes:
    """
    Serialize a single layer's KV cache to bytes.

    Args:
        kv_cache_layer: MLX KV cache object for one layer

    Returns:
        Serialized bytes containing K and V tensors
    """
    # MLX KV cache has keys and values as mx.array
    # Handle different cache types (QuantizedKVCache, etc.)
    if hasattr(kv_cache_layer, "keys") and hasattr(kv_cache_layer, "values"):
        k = np.array(kv_cache_layer.keys, copy=False)
        v = np.array(kv_cache_layer.values, copy=False)
    elif hasattr(kv_cache_layer, "state"):
        # Some caches store state as tuple (k, v)
        k = np.array(kv_cache_layer.state[0], copy=False)
        v = np.array(kv_cache_layer.state[1], copy=False)
    else:
        # Fallback: assume it's indexable
        k = np.array(kv_cache_layer[0], copy=False)
        v = np.array(kv_cache_layer[1], copy=False)

    # Pack with shape info
    k_flat = k.reshape(-1).astype(np.float16)
    v_flat = v.reshape(-1).astype(np.float16)

    header = np.array(
        [
            len(k.shape),
            *k.shape,
            len(v.shape),
            *v.shape,
        ],
        dtype=np.int32,
    )

    return header.tobytes() + k_flat.tobytes() + v_flat.tobytes()


def deserialize_kv_layer(data: bytes) -> tuple[mx.array, mx.array]:
    """
    Deserialize bytes back to K, V tensors.

    Returns:
        Tuple of (keys, values) as mx.array
    """
    # Read header
    header_count = 0
    idx = 0

    # Read K shape
    k_ndim = int(np.frombuffer(data[idx : idx + 4], dtype=np.int32)[0])
    idx += 4
    header_count += 1

    k_shape = tuple(
        np.frombuffer(data[idx : idx + 4 * k_ndim], dtype=np.int32).tolist()
    )
    idx += 4 * k_ndim

    # Read V shape
    v_ndim = int(np.frombuffer(data[idx : idx + 4], dtype=np.int32)[0])
    idx += 4

    v_shape = tuple(
        np.frombuffer(data[idx : idx + 4 * v_ndim], dtype=np.int32).tolist()
    )
    idx += 4 * v_ndim

    # Read K data
    k_size = int(np.prod(k_shape))
    k_flat = np.frombuffer(data[idx : idx + k_size * 2], dtype=np.float16)
    idx += k_size * 2
    k = mx.array(k_flat.reshape(k_shape))

    # Read V data
    v_size = int(np.prod(v_shape))
    v_flat = np.frombuffer(data[idx : idx + v_size * 2], dtype=np.float16)
    v = mx.array(v_flat.reshape(v_shape))

    return k, v


async def allgather_ring(
    local_data: bytes,
    ring_comm: CPRingCommunicator,
    tag_prefix: str,
) -> list[bytes]:
    """
    AllGather via ring: collect data from all ranks.

    Uses N-1 ring rotations to gather all chunks.

    Args:
        local_data: This rank's data
        ring_comm: Ring communicator
        tag_prefix: Unique tag prefix for this gather

    Returns:
        List of data from all ranks, in rank order
    """
    num_ranks = ring_comm.num_ranks
    rank_id = ring_comm.rank_id

    if num_ranks == 1:
        return [local_data]

    # Storage for all chunks, indexed by original rank
    all_chunks: list[Optional[bytes]] = [None] * num_ranks
    all_chunks[rank_id] = local_data

    # Current chunk to send (starts as ours, then becomes received)
    current_chunk = local_data
    source_rank = rank_id

    for step in range(1, num_ranks):
        tag = f"{tag_prefix}_step{step}"

        # Ring send/recv: send current to next, receive from prev
        recv_chunk = await ring_comm.send_recv(current_chunk, tag)

        # Calculate which rank's data we received
        source_rank = (source_rank - 1) % num_ranks
        all_chunks[source_rank] = recv_chunk

        # Next iteration: forward what we received
        current_chunk = recv_chunk

    return [c for c in all_chunks if c is not None]


async def sync_kv_cache_layer(
    kv_cache_layer,
    layer_idx: int,
    ring_comm: CPRingCommunicator,
    nonce: str,
) -> None:
    """
    Synchronize a single layer's KV cache across all CP ranks.

    After this call, each rank has KV from all ranks concatenated.

    Args:
        kv_cache_layer: The KV cache object for this layer
        layer_idx: Layer index (for logging)
        ring_comm: Ring communicator
        nonce: Request nonce (for unique tags)
    """
    if ring_comm.num_ranks == 1:
        return

    # Serialize local KV
    local_kv_bytes = serialize_kv_layer(kv_cache_layer)

    # AllGather KV from all ranks
    all_kv_bytes = await allgather_ring(
        local_kv_bytes,
        ring_comm,
        f"kv_L{layer_idx}_{nonce[:8]}",
    )

    # Deserialize all chunks
    all_kvs = [deserialize_kv_layer(b) for b in all_kv_bytes]

    # Concatenate along sequence dimension (axis 2 for [B, H, S, D])
    all_keys = [kv[0] for kv in all_kvs]
    all_values = [kv[1] for kv in all_kvs]

    merged_k = mx.concatenate(all_keys, axis=2)
    merged_v = mx.concatenate(all_values, axis=2)

    # Update the cache in-place
    if hasattr(kv_cache_layer, "keys") and hasattr(kv_cache_layer, "values"):
        kv_cache_layer.keys = merged_k
        kv_cache_layer.values = merged_v
    elif hasattr(kv_cache_layer, "state"):
        kv_cache_layer.state = (merged_k, merged_v)

    logger.debug(
        "CP sync layer %d: %d ranks -> merged KV shape %s",
        layer_idx,
        ring_comm.num_ranks,
        merged_k.shape,
    )


async def sync_full_kv_cache(
    kv_cache: list,
    ring_comm: CPRingCommunicator,
    nonce: str,
) -> None:
    """
    Synchronize all layers' KV caches across CP ranks.

    Calls sync_kv_cache_layer for each layer in parallel.
    """
    if ring_comm.num_ranks == 1:
        return

    tasks = [
        sync_kv_cache_layer(kv_cache[i], i, ring_comm, nonce)
        for i in range(len(kv_cache))
    ]
    await asyncio.gather(*tasks)
