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
        # Order devices by Thunderbolt connectivity for minimal latency
        ordered_instances = self._optimize_ring_order(
            profiles, thunderbolts, list(shards.keys())
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
    """API adapter for context parallel communication."""

    def __init__(self) -> None:
        super().__init__()
        # For CP, we broadcast tokens to all shards (rank 0 is primary)
        self.primary_channel: Optional[aio_grpc.Channel] = None
        self.primary_stub: Optional[DnetRingServiceStub] = None
        self._streams = StreamManager(idle_timeout_s=5.0, backoff_s=0.2)
        self._pending: Dict[str, asyncio.Future[TokenResult]] = {}

    async def start(self) -> None:
        self.running = True

    async def shutdown(self) -> None:
        self.running = False
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

    async def connect_first_shard(self, ip: str, port: int) -> None:
        """Connect to primary shard (rank 0) which coordinates CP."""
        target = f"{ip}:{port}"
        if self.primary_channel:
            try:
                await self.primary_channel.close()
            except Exception:
                pass
        self.primary_channel = aio_grpc.insecure_channel(target)
        self.primary_stub = DnetRingServiceStub(self.primary_channel)
        logger.info("CP adapter connected to primary shard at %s", target)

    async def reset_cache(self) -> None:
        if not self.primary_stub:
            raise RuntimeError("CP adapter not connected")
        try:
            await self.primary_stub.ResetCache(pb2.ResetCacheRequest())
        except Exception as e:
            logger.warning("ResetCache RPC failed: %s", e)

    async def send_tokens(
        self,
        nonce: str,
        tokens: bytes,
        callback_addr: str,
        logprobs: bool = False,
        top_logprobs: int = 0,
        decoding_config: Optional[Any] = None,
    ) -> None:
        """Send tokens to primary shard for CP inference."""
        if not self.primary_stub:
            raise RuntimeError("CP adapter not connected to primary shard")

        msg = ActivationMessage(
            nonce=nonce,
            pool_id=-1,
            batch_size=1,
            shape=(1,),
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
