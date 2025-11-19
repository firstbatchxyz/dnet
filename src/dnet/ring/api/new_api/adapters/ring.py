from __future__ import annotations

import asyncio
from typing import Dict, Optional, Any
from grpc import aio as aio_grpc

import numpy as np

from .....utils.logger import logger
from .....core.stream_manager import StreamManager
from dnet.protos import dnet_ring_pb2 as pb2
from dnet.protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from .base import ApiAdapterBase


class ApiRingAdapter(ApiAdapterBase):
    def __init__(self) -> None:
        super().__init__()
        self.channel: Optional[aio_grpc.Channel] = None
        self.stub: Optional[DnetRingServiceStub] = None
        self._streams = StreamManager(idle_timeout_s=5.0, backoff_s=0.2)
        self._pending: Dict[str, asyncio.Future[int]] = {}

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
            await self.stub.ResetCache(pb2.ResetCacheRequest())  # type: ignore[arg-type]
        except Exception as e:
            logger.warning("ResetCache RPC failed: %s", e)

    async def send_tokens(self, nonce: str, tokens: bytes, callback_addr: str) -> None:
        if not self.stub:
            raise RuntimeError("API adapter not connected to a shard")

        # Build activation request carrying tokens
        n_tokens = int(len(tokens) // np.dtype(np.int32).itemsize)
        act = pb2.Activation(
            batch_size=1,
            shape=[n_tokens],
            dtype="tokens",
            layer_id=-1,
            data=tokens,
        )
        req = pb2.ActivationRequest(
            nonce=nonce,
            node_origin="api",
            timestamp=0,
            activation=act,
            callback_url=f"grpc://{callback_addr}",
        )

        # Stream via StreamActivations to mirror shard behavior
        ctx = await self._streams.get_or_create_stream(
            nonce,
            lambda it: self.stub.StreamActivations(it),  # type: ignore[attr-defined]
        )
        if not ctx or not ctx.open or ctx.disabled:
            raise RuntimeError("API stream not available for nonce %s" % nonce)
        ctx.last_seq += 1
        await ctx.queue.put(
            pb2.ActivationFrame(request=req, seq=ctx.last_seq, end_of_request=False)
        )
        ctx.last_activity_t = asyncio.get_running_loop().time()

    async def await_token(self, nonce: str, timeout_s: float) -> int:
        fut = asyncio.get_running_loop().create_future()
        self._pending[nonce] = fut
        try:
            return int(await asyncio.wait_for(fut, timeout=timeout_s))
        finally:
            self._pending.pop(nonce, None)

    def resolve_token(self, nonce: str, token_id: int) -> None:
        fut = self._pending.get(nonce)
        if fut and not fut.done():
            fut.set_result(int(token_id))

