from __future__ import annotations
import asyncio
from typing import Dict, Optional, Any, Literal
from grpc import aio as aio_grpc

from dnet_p2p import DnetDeviceProperties, ThunderboltConnection
from distilp.common import DeviceProfile
from distilp.solver import halda_solve, HALDAResult

from dnet.utils.logger import logger
from dnet.core.stream_manager import StreamManager
from dnet.utils.time import utc_epoch_now
from dnet.core.types.messages import ActivationMessage
from dnet.protos import dnet_ring_pb2 as pb2
from dnet.protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from dnet.core.types.topology import TopologyInfo
from ..utils import (
    optimize_device_ordering,
    compute_layer_assignments,
    postprocess_single_round,
)
from .base import Strategy, ApiAdapterBase
from dnet.core.types.messages import TokenResult
from dnet.core.topology import TopologySolver


class RingTopologySolver(TopologySolver):
    """
    Topology solver using the HALDA MILP algorithm for ring topology.
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
        # 1. Optimize ordering (specific to ring topology)
        ordered_instances = optimize_device_ordering(profiles, thunderbolts)

        # 2. Prepare profiles for solver
        sorted_shard_profiles = [
            profiles[name] for name in ordered_instances if name in profiles
        ]
        if not sorted_shard_profiles:
            raise ValueError("No valid shard profiles found")

        # mark the first device as head, others as non-head
        for i, profile in enumerate(sorted_shard_profiles):
            profile.is_head = i == 0

        logger.info("Running solver with %d shard profiles", len(sorted_shard_profiles))

        # 3. Run solver
        solution: HALDAResult = halda_solve(
            devs=sorted_shard_profiles,
            model=model_profile,
            mip_gap=1e-4,
            plot=False,
            kv_bits=kv_bits,
        )

        logger.info(
            "Solver completed: k=%d, objective=%d", solution.k, solution.obj_value
        )

        # 4. Post-process solution
        ordered_instances, solution = postprocess_single_round(
            ordered_instances, solution
        )

        # 5. Compute assignments
        layer_assignments = compute_layer_assignments(
            ordered_instances,
            solution.w,
            solution.n,
            solution.k,
        )

        shards_list = [shards[name] for name in ordered_instances]

        # 6. Create TopologyInfo
        return TopologyInfo(
            model=model_name,
            kv_bits=kv_bits,
            num_layers=num_layers,
            devices=shards_list,
            assignments=layer_assignments,
            solution=solution,
        )


class RingApiAdapter(ApiAdapterBase):
    def __init__(self) -> None:
        super().__init__()
        self.channel: Optional[aio_grpc.Channel] = None
        self.stub: Optional[DnetRingServiceStub] = None
        self._streams = StreamManager(idle_timeout_s=5.0, backoff_s=0.2)
        self._pending: Dict[str, asyncio.Future[TokenResult]] = {}

    async def start(self) -> None:
        self.running = True

    async def shutdown(self) -> None:
        self.running = False
        # end any open streams
        for nonce in list(getattr(self._streams, "_streams", {}).keys()):
            try:
                await self._streams.end_stream(nonce)
            except Exception:
                pass
        # close channel
        try:
            if self.channel:
                await self.channel.close()
        except Exception:
            pass
        self.channel = None
        self.stub = None

    async def connect_first_shard(self, ip: str, port: int) -> None:
        target = f"{ip}:{port}"
        if self.channel:
            try:
                await self.channel.close()
            except Exception:
                pass
        self.channel = aio_grpc.insecure_channel(target)
        self.stub = DnetRingServiceStub(self.channel)
        logger.info("Connected API adapter to first shard at %s", target)

    async def reset_cache(self) -> None:
        if not self.stub:
            raise RuntimeError("API adapter not connected to a shard")
        try:
            await self.stub.ResetCache(pb2.ResetCacheRequest())
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
        if not self.stub:
            raise RuntimeError("Ring adapter not connected to first shard")

        msg = ActivationMessage(
            nonce=nonce,
            pool_id=-1,
            batch_size=1,
            shape=(1,),  # dummy shape for tokens
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
            repetition_penalty=decoding_config.repetition_penalty
            if decoding_config
            else 1.0,
            min_p=decoding_config.min_p if decoding_config else 0.0,
            min_tokens_to_keep=decoding_config.min_tokens_to_keep
            if decoding_config
            else 1,
        )
        req = msg.to_proto(tokens)

        if self.stub is None:
            raise RuntimeError("Ring adapter not connected to first shard")
        stub = self.stub

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


class RingStrategy(Strategy):
    def __init__(self):
        self._solver = RingTopologySolver()
        self._adapter = RingApiAdapter()

    @property
    def solver(self) -> TopologySolver:
        return self._solver

    @property
    def adapter(self) -> ApiAdapterBase:
        return self._adapter
