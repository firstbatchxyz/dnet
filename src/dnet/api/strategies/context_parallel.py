"""Context Parallel strategy for API server.

This module provides the ContextParallelStrategy which bundles:
- CPTopologySolver: Assigns all layers to all devices (full replication)
- CPApiAdapter: Handles token injection for CP mode
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional, Any, Literal, List

from grpc import aio as aio_grpc
from dnet_p2p import DnetDeviceProperties, ThunderboltConnection
from distilp.common import DeviceProfile

from dnet.utils.logger import logger
from dnet.core.stream_manager import StreamManager
from dnet.core.types.messages import TokenResult
from dnet.core.types.topology import TopologyInfo, LayerAssignment
from dnet.core.topology import TopologySolver
from dnet.protos import dnet_ring_pb2 as pb2
from dnet.protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from dnet.utils.time import utc_epoch_now
from dnet.core.types.messages import ActivationMessage
from dnet.core.cp.sharding import shard_for_mode
from .base import Strategy, ApiAdapterBase


class CPTopologyInfo(TopologyInfo):
    """Extended topology info for context parallelism."""

    num_cp_ranks: int = 1
    cp_algorithm: str = "auto"


class CPTopologySolver(TopologySolver):
    """
    Topology solver for context parallelism.

    Unlike ring topology, CP assigns ALL layers to EACH device.
    Optimization focuses on ordering devices for minimal ring latency.
    """

    async def solve(
        self,
        profiles: Dict[str, DeviceProfile],
        model_profile: Any,
        model_name: str,
        num_layers: int,
        kv_bits: Literal["4bit", "8bit", "fp16"],
        shards: Dict[str, DnetDeviceProperties],
        thunderbolts: Dict[str, Dict[str, ThunderboltConnection]],
    ) -> TopologyInfo:
        """
        Solve topology for context parallelism.

        For CP, all devices get the full model. We optimize the ring
        ordering for minimal inter-device latency.
        """

        # Filter out manager nodes - only include actual shards that have profiles
        active_shards = {
            name: props
            for name, props in shards.items()
            if not props.is_manager and name in profiles
        }

        # Order devices by Thunderbolt connectivity for minimal latency
        ordered_instances = self._optimize_ring_order(
            profiles, thunderbolts, list(active_shards.keys())
        )

        # Build layer assignments as list of LayerAssignment objects
        # For CP, each device gets ALL layers (full model replication)
        all_layers = list(range(num_layers))
        layer_assignments: List[LayerAssignment] = []

        for i, name in enumerate(ordered_instances):
            next_name = (
                ordered_instances[(i + 1) % len(ordered_instances)]
                if len(ordered_instances) > 1
                else None
            )
            layer_assignments.append(
                LayerAssignment(
                    instance=name,
                    layers=[all_layers],  # All layers in single round (k=1)
                    next_instance=next_name,
                    window_size=num_layers,
                    residency_size=num_layers,
                )
            )

        shards_list = [shards[name] for name in ordered_instances]

        logger.info(
            "CP topology: %d devices, each with all %d layers",
            len(ordered_instances),
            num_layers,
        )

        # Create TopologyInfo
        return TopologyInfo(
            model=model_name,
            kv_bits=kv_bits,
            num_layers=num_layers,
            devices=shards_list,
            assignments=layer_assignments,
            solution=None,  # No HALDA solution for CP
        )

    def _optimize_ring_order(
        self,
        profiles: Dict[str, DeviceProfile],
        thunderbolts: Dict[str, Dict[str, ThunderboltConnection]],
        device_names: list[str],
    ) -> list[str]:
        """
        Order devices to minimize ring latency.

        Prioritize Thunderbolt connections, fallback to device order.
        """
        if len(device_names) <= 2:
            return device_names

        # Build adjacency matrix of TB connections
        has_tb = {}
        for src in device_names:
            if src in thunderbolts:
                for dst, conn in thunderbolts[src].items():
                    if dst in device_names and conn.ip_addr:
                        has_tb[(src, dst)] = True

        # Greedy ordering: start from first, pick next with TB if possible
        ordered = [device_names[0]]
        remaining = set(device_names[1:])

        while remaining:
            current = ordered[-1]
            # Find neighbor with TB connection
            next_device = None
            for candidate in remaining:
                if has_tb.get((current, candidate)):
                    next_device = candidate
                    break

            if not next_device:
                # No TB connection, pick arbitrary
                next_device = remaining.pop()
            else:
                remaining.remove(next_device)

            ordered.append(next_device)

        return ordered


class CPApiAdapter(ApiAdapterBase):
    """API adapter for context parallel communication.

    Supports multi-rank broadcasting: splits token sequence across ranks
    and sends chunks in parallel. Only the last rank samples and returns.
    """

    def __init__(self) -> None:
        super().__init__()
        # Legacy single-shard connection (kept for backward compat)
        self.primary_channel: Optional[aio_grpc.Channel] = None
        self.primary_stub: Optional[DnetRingServiceStub] = None
        self._streams = StreamManager(idle_timeout_s=5.0, backoff_s=0.2)
        self._pending: Dict[str, asyncio.Future[TokenResult]] = {}

        # Multi-rank connections for CP
        self.num_ranks: int = 1
        self.rank_channels: Dict[int, aio_grpc.Channel] = {}
        self.rank_stubs: Dict[int, DnetRingServiceStub] = {}
        self._streams_by_rank: Dict[int, StreamManager] = {}

    async def start(self) -> None:
        self.running = True

    async def shutdown(self) -> None:
        self.running = False
        # Clean up legacy streams
        for nonce in list(getattr(self._streams, "_streams", {}).keys()):
            try:
                await self._streams.end_stream(nonce)
            except Exception:
                pass
        if self.primary_channel:
            try:
                await self.primary_channel.close()
            except Exception:
                pass
        self.primary_channel = None
        self.primary_stub = None

        # Clean up multi-rank streams and channels
        for streams in self._streams_by_rank.values():
            for nonce in list(getattr(streams, "_streams", {}).keys()):
                try:
                    await streams.end_stream(nonce)
                except Exception:
                    pass
        for channel in self.rank_channels.values():
            try:
                await channel.close()
            except Exception:
                pass
        self.rank_channels.clear()
        self.rank_stubs.clear()
        self._streams_by_rank.clear()

    async def connect_first_shard(self, ip: str, port: int) -> None:
        """Connect to primary shard (rank 0) - legacy single-shard mode."""
        target = f"{ip}:{port}"
        if self.primary_channel:
            try:
                await self.primary_channel.close()
            except Exception:
                pass
        from dnet.utils.grpc_config import GRPC_AIO_OPTIONS

        self.primary_channel = aio_grpc.insecure_channel(
            target, options=GRPC_AIO_OPTIONS
        )
        self.primary_stub = DnetRingServiceStub(self.primary_channel)
        logger.info("CP adapter connected to primary shard at %s", target)

    async def connect_all_ranks(self, rank_addresses: List[str]) -> None:
        """Connect to all CP ranks for multi-rank broadcasting.

        Args:
            rank_addresses: List of "host:port" strings, one per rank, in order.
        """
        from dnet.utils.grpc_config import GRPC_AIO_OPTIONS

        # Close existing connections
        for channel in self.rank_channels.values():
            try:
                await channel.close()
            except Exception:
                pass
        self.rank_channels.clear()
        self.rank_stubs.clear()
        self._streams_by_rank.clear()

        self.num_ranks = len(rank_addresses)
        for rank, addr in enumerate(rank_addresses):
            self.rank_channels[rank] = aio_grpc.insecure_channel(
                addr, options=GRPC_AIO_OPTIONS
            )
            self.rank_stubs[rank] = DnetRingServiceStub(self.rank_channels[rank])
            self._streams_by_rank[rank] = StreamManager(
                idle_timeout_s=60.0, backoff_s=0.2
            )

        # Also set primary for backward compat
        if rank_addresses:
            self.primary_channel = self.rank_channels.get(0)
            self.primary_stub = self.rank_stubs.get(0)

        logger.info(
            "CP adapter connected to %d ranks: %s", self.num_ranks, rank_addresses
        )

    async def reset_cache(self) -> None:
        """Reset cache on all ranks."""
        if self.num_ranks > 1 and self.rank_stubs:
            # Multi-rank: reset on all
            async def reset_rank(rank: int):
                stub = self.rank_stubs.get(rank)
                if stub:
                    try:
                        await stub.ResetCache(pb2.ResetCacheRequest())
                    except Exception as e:
                        logger.warning("ResetCache failed on rank %d: %s", rank, e)

            await asyncio.gather(*[reset_rank(r) for r in range(self.num_ranks)])
        elif self.primary_stub:
            # Single-rank fallback
            try:
                await self.primary_stub.ResetCache(pb2.ResetCacheRequest())
            except Exception as e:
                logger.warning("ResetCache RPC failed: %s", e)
        else:
            raise RuntimeError("CP adapter not connected")

    async def send_tokens(
        self,
        nonce: str,
        tokens: bytes,
        callback_addr: str,
        logprobs: bool = False,
        top_logprobs: int = 0,
        decoding_config: Optional[Any] = None,
        start_pos: int = 0,
    ) -> None:
        """Send tokens to all CP ranks (split and broadcast).

        If multi-rank is configured, splits the token sequence using
        shard_for_mode() and sends each chunk to its corresponding rank.
        Only the last rank will sample and return the result.
        """
        if self.num_ranks > 1 and self.rank_stubs:
            # Multi-rank mode: split and broadcast
            await self._send_tokens_multi_rank(
                nonce,
                tokens,
                callback_addr,
                logprobs,
                top_logprobs,
                decoding_config,
                start_pos,
            )
        elif self.primary_stub:
            # Single-rank fallback (legacy behavior)
            await self._send_tokens_single_rank(
                nonce, tokens, callback_addr, logprobs, top_logprobs, decoding_config
            )
        else:
            raise RuntimeError("CP adapter not connected to any shard")

    async def _send_tokens_single_rank(
        self,
        nonce: str,
        tokens: bytes,
        callback_addr: str,
        logprobs: bool,
        top_logprobs: int,
        decoding_config: Optional[Any],
    ) -> None:
        """Legacy single-rank send (original behavior)."""
        msg = ActivationMessage(
            nonce=nonce,
            pool_id=-1,
            batch_size=1,
            shape=(len(tokens) // 4,),  # int32 tokens
            dtype="tokens",
            layer_id=-1,
            timestamp=utc_epoch_now(),
            node_origin="api",
            callback_url=f"grpc://{callback_addr}",
            req_logprobs=logprobs,
            req_top_logprobs=top_logprobs,
            temperature=decoding_config.temperature if decoding_config else 1.0,
            top_p=decoding_config.top_p if decoding_config else 1.0,
            top_k=decoding_config.top_k if decoding_config else -1,
            repetition_penalty=(
                decoding_config.repetition_penalty if decoding_config else 1.0
            ),
            min_p=decoding_config.min_p if decoding_config else 0.0,
            min_tokens_to_keep=(
                decoding_config.min_tokens_to_keep if decoding_config else 1
            ),
        )
        req = msg.to_proto(tokens)

        stub = self.primary_stub
        assert stub is not None, "primary_stub should be set"
        ctx = await self._streams.get_or_create_stream(
            nonce,
            lambda it: stub.StreamActivations(it),
        )
        if not ctx or not ctx.open:
            raise RuntimeError(f"Failed to create stream for nonce {nonce}")

        ctx.last_seq += 1
        await ctx.queue.put(
            pb2.ActivationFrame(request=req, seq=ctx.last_seq, end_of_request=False)
        )
        ctx.last_activity_t = asyncio.get_running_loop().time()

    async def _send_tokens_multi_rank(
        self,
        nonce: str,
        tokens: bytes,
        callback_addr: str,
        logprobs: bool,
        top_logprobs: int,
        decoding_config: Optional[Any],
        start_pos: int,
    ) -> None:
        """Multi-rank send: broadcast full tokens to all ranks for Ring Attention."""
        import numpy as np

        # Deserialize full token sequence
        full_tokens = np.frombuffer(tokens, dtype=np.int32)
        num_tokens = len(full_tokens)

        logger.debug(
            "CP multi-rank send: nonce=%s, %d tokens -> %d ranks",
            nonce,
            num_tokens,
            self.num_ranks,
        )

        # For decode (single token), send to ALL ranks (Broadcast).
        # Each rank needs the full Q to attend to its local KV shard.
        if num_tokens == 1:

            async def send_broadcast(rank: int) -> None:
                # Only the last rank should sample/generate tokens
                is_last_rank = rank == self.num_ranks - 1

                await self._send_chunk_to_rank(
                    rank,
                    nonce,
                    tokens,  # Full tokens (broadcast)
                    callback_addr,
                    logprobs if is_last_rank else False,
                    top_logprobs if is_last_rank else 0,
                    decoding_config if is_last_rank else None,
                    num_tokens,
                    rope_offset=start_pos,
                )

            await asyncio.gather(*[send_broadcast(r) for r in range(self.num_ranks)])
            return

        # Phase 5: True Ring Attention (Sharded KV)
        # Use load-balanced 2N sharding for prefill to ensure each rank stores only 1/N KV.
        # The CPAttentionWrapper will use CPAdapter to rotate KV blocks.

        # Helper to send sharded chunk to a rank
        async def send_shard_to_rank(rank: int) -> None:
            import mlx.core as mx
            import numpy as np

            # shard_for_mode expects mx.array, convert from numpy
            mx_tokens = mx.array(full_tokens)

            # Get shard for this rank (prefill mode)
            sharded_chunk_mx, indices = shard_for_mode(
                mx_tokens, self.num_ranks, rank, "prefill"
            )

            # Convert back to bytes for network transmission
            # mx.array -> numpy -> bytes
            chunk_np = np.array(sharded_chunk_mx)
            chunk_bytes = chunk_np.tobytes()

            # Only the last rank should sample/generate tokens
            is_last_rank = rank == self.num_ranks - 1

            # Use existing send helper
            # RoPE offset is globally determined by the start index of this shard
            chunk_offset = start_pos + indices[0] if indices else start_pos

            await self._send_chunk_to_rank(
                rank,
                nonce,
                chunk_bytes,
                callback_addr,
                logprobs if is_last_rank else False,
                top_logprobs if is_last_rank else 0,
                decoding_config if is_last_rank else None,
                len(chunk_np),
                rope_offset=chunk_offset,
            )

        # Send sharded chunks to all ranks in parallel
        await asyncio.gather(*[send_shard_to_rank(r) for r in range(self.num_ranks)])

    async def _send_chunk_to_rank(
        self,
        rank: int,
        nonce: str,
        tokens: bytes,
        callback_addr: str,
        logprobs: bool,
        top_logprobs: int,
        decoding_config: Optional[Any],
        num_tokens: int,
        rope_offset: int,
    ) -> None:
        """Send tokens directly to a specific rank (for decode phase)."""
        logger.debug(
            "CP decode: sending %d tokens directly to rank %d (last rank)",
            num_tokens,
            rank,
        )

        msg = ActivationMessage(
            nonce=nonce,
            pool_id=-1,
            batch_size=1,
            shape=(num_tokens,),
            dtype="tokens",
            layer_id=-1,
            timestamp=utc_epoch_now(),
            node_origin="api",
            callback_url=f"grpc://{callback_addr}",
            req_logprobs=logprobs,
            req_top_logprobs=top_logprobs,
            temperature=decoding_config.temperature if decoding_config else 1.0,
            top_p=decoding_config.top_p if decoding_config else 1.0,
            top_k=decoding_config.top_k if decoding_config else -1,
            repetition_penalty=(
                decoding_config.repetition_penalty if decoding_config else 1.0
            ),
            min_p=decoding_config.min_p if decoding_config else 0.0,
            min_tokens_to_keep=(
                decoding_config.min_tokens_to_keep if decoding_config else 1
            ),
            rope_offset=rope_offset,
        )
        req = msg.to_proto(tokens)

        stub = self.rank_stubs[rank]
        streams = self._streams_by_rank[rank]
        ctx = await streams.get_or_create_stream(
            nonce,
            lambda it: stub.StreamActivations(it),
        )
        if not ctx or not ctx.open:
            raise RuntimeError(
                f"Failed to create stream for rank {rank}, nonce {nonce}"
            )

        ctx.last_seq += 1
        await ctx.queue.put(
            pb2.ActivationFrame(request=req, seq=ctx.last_seq, end_of_request=False)
        )
        ctx.last_activity_t = asyncio.get_running_loop().time()

    async def await_token(self, nonce: str, timeout_s: float) -> TokenResult:
        fut = asyncio.get_running_loop().create_future()
        self._pending[nonce] = fut
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        finally:
            self._pending.pop(nonce, None)

    def resolve_token(self, nonce: str, result: TokenResult) -> None:
        fut = self._pending.get(nonce)
        if fut and not fut.done():
            fut.set_result(result)


class ContextParallelStrategy(Strategy):
    """
    Execution strategy using context parallelism.

    Distributes sequence dimension across devices while replicating
    all model layers on each device.
    """

    def __init__(self):
        self._solver = CPTopologySolver()
        self._adapter = CPApiAdapter()

    @property
    def solver(self) -> TopologySolver:
        return self._solver

    @property
    def adapter(self) -> ApiAdapterBase:
        return self._adapter
