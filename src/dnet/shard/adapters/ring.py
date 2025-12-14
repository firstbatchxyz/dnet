"""
RingAdapter: ring transport + topology glue around a topology‑agnostic runtime.

- Ingress: receive ActivationRequest, stage payloads, enqueue to runtime
- Egress: read ActivationMessage from runtime and stream to next node or API
- Streaming only: no unary fallback to keep logic simple and consistent
"""

from __future__ import annotations
import queue
from typing import Optional
import asyncio
import time
from ..models import ShardLoadModelRequest
from dnet_p2p import (
    AsyncDnetP2P,
    DnetDeviceProperties,
    discover_thunderbolt_connection,
)
from grpc import aio as aio_grpc
from urllib.parse import urlparse

from dnet.utils.grpc_config import GRPC_AIO_OPTIONS
from dnet.utils.time import utc_epoch_now
from dnet.protos import shard_api_comm_pb2, shard_api_comm_pb2_grpc
from .base import TopologyAdapter
from ..runtime import ShardRuntime
from dnet.protos.dnet_ring_pb2 import ActivationRequest
from dnet.protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from dnet.core.types.messages import ActivationMessage
from dnet.utils.logger import logger
from dnet.core.stream_manager import StreamManager
from dnet.config import get_settings, TransportSettings
from dnet.protos import dnet_ring_pb2 as pb2
from ..codec import ActivationCodec
from dnet.protos.shard_api_comm_pb2_grpc import ShardApiServiceStub


class RingAdapter(TopologyAdapter):
    def __init__(
        self,
        runtime: ShardRuntime,
        discovery: AsyncDnetP2P,
        transport_settings: Optional[TransportSettings] = None,
    ) -> None:
        super().__init__(runtime, discovery)

        # Use provided settings or load from centralized config
        self.transport_settings: TransportSettings = (
            transport_settings if transport_settings else get_settings().transport
        )
        self.codec = ActivationCodec(runtime)

        self.running = False
        self._active_nonce: Optional[str] = None

        self._streaming_enabled = self.transport_settings.streaming
        self._streams: StreamManager = StreamManager(
            idle_timeout_s=self.transport_settings.stream_idle_s,
            backoff_s=self.transport_settings.stream_backoff_s,
        )

        # Topology
        self.next_node: Optional[DnetDeviceProperties] = None
        self.next_node_channel: Optional[aio_grpc.Channel] = None
        self.next_node_stub: Optional[DnetRingServiceStub] = None

        # Implement the required queues
        self.queue_size = runtime.max_queue_size
        self._ingress_q: asyncio.Queue[ActivationRequest] = asyncio.Queue(
            maxsize=self.queue_size
        )
        self.ring_tx_q: asyncio.Queue[ActivationMessage] = asyncio.Queue(
            maxsize=self.queue_size
        )
        self.token_tx_q: asyncio.Queue[ActivationMessage] = asyncio.Queue(
            maxsize=self.queue_size
        )

        # API callback gRPC
        self.api_channel: Optional[aio_grpc.Channel] = None
        self.api_stub: Optional[ShardApiServiceStub] = None
        self.api_address: Optional[str] = None

        self.total_layers: int = 0
        self.api_callback_address: Optional[str] = None

    async def start(self):
        self.running = True
        loop = asyncio.get_running_loop()
        self._loop = loop  # if you need it in workers
        self._tasks = [
            asyncio.create_task(self._ingress_worker()),
            asyncio.create_task(self._egress_worker()),
            asyncio.create_task(self._ring_tx_worker()),
            asyncio.create_task(self._token_tx_worker()),
        ]
        if self._streaming_enabled:
            self._tasks.append(asyncio.create_task(self._stream_sweeper()))

    @property
    def ingress_q(self) -> asyncio.Queue:
        return self._ingress_q

    @property
    def activation_computed_queue(self) -> asyncio.Queue:
        return self.ring_tx_q

    @property
    def activation_token_queue(self) -> asyncio.Queue:
        return self.token_tx_q

    async def ingress(self):
        pass

    async def egress(self):
        pass

    async def configure_topology(self, req: ShardLoadModelRequest):
        self.next_node = req.next_node
        self.total_layers = req.total_layers
        self.api_callback_address = req.api_callback_address

        if self.next_node:
            await self._connect_next_node()
        else:
            logger.warning("Node %s: No next node configured", self.runtime.shard_id)

    async def reset_topology(self):
        """Reset topology state."""
        self.next_node = None
        self.total_layers = 0
        self.api_callback_address = None

        if self.next_node_channel:
            try:
                await self.next_node_channel.close()
            except Exception:
                pass
            self.next_node_channel = None
            self.next_node_stub = None

        if self.api_channel:
            try:
                await self.api_channel.close()
            except Exception:
                pass
            self.api_channel = None
            self.api_stub = None
            self.api_address = None

    async def admit_frame(self, request: ActivationRequest) -> None:
        while self.running:
            try:
                self.ingress_q.put_nowait(request)
                return
            except asyncio.QueueFull:
                await asyncio.sleep(0)
        return

    async def _ingress_worker(self):
        """Drains ingress queue and processes frames with heavy work offloaded."""
        loop = asyncio.get_running_loop()

        while self.running:
            try:
                req = await self.ingress_q.get()
            except asyncio.CancelledError:
                break
            try:
                await self._connect_next_node()

                activation = req.activation
                target_layer = activation.layer_id + 1

                # Detect new sequence per node: initialize per-nonce KV
                # TODO: replace it with a helper function
                if req.nonce != self._active_nonce:
                    self._active_nonce = req.nonce
                    self.runtime.get_or_make_kv(req.nonce)

                if target_layer in self.runtime._assigned_set:
                    # Heavy prep in executor (alloc/copy/decompress)
                    try:
                        activation_msg = await loop.run_in_executor(
                            self.runtime.executor,
                            self.codec.deserialize,
                            req,
                        )
                    except Exception as e:
                        logger.error(
                            "Codec deserialize failed for nonce %s: %s", req.nonce, e
                        )
                        continue

                    if activation_msg:
                        await loop.run_in_executor(
                            None,
                            self.runtime.activation_recv_queue.put_nowait,
                            activation_msg,
                        )
                else:
                    await self._forward_activation(req)

            except Exception as e:
                logger.error("Ingress worker error: %s", e)

    async def _egress_worker(self):
        loop = asyncio.get_running_loop()
        q = self.runtime.activation_send_queue

        while self.running:
            try:
                msg = await loop.run_in_executor(
                    self.runtime.executor,
                    lambda: q.get(timeout=0.5),
                )
            except asyncio.CancelledError:
                break
            except queue.Empty:
                continue

            target = self.token_tx_q if msg.is_final else self.ring_tx_q
            await target.put(msg)

    async def _ring_tx_worker(self):
        while self.running or not self.ring_tx_q.empty():
            msg = await self.ring_tx_q.get()
            await self._send_activation(msg)

    async def _token_tx_worker(self):
        while self.running or not self.token_tx_q.empty():
            msg = await self.token_tx_q.get()
            await self._send_token(msg)

    async def _stream_sweeper(self):
        while self.running:
            await self._streams.cleanup_idle_streams()
            await asyncio.sleep(1.0)

    async def _forward_activation(self, request: ActivationRequest):
        if not (self._streaming_enabled and self.next_node_stub):
            logger.error(
                "Streaming disabled or next node not connected; cannot forward"
            )
            return

        if self.next_node_stub is None:
            raise ValueError("next_node_stub is None")
        stub = self.next_node_stub

        ctx = await self._streams.get_or_create_stream(
            request.nonce,
            lambda it: stub.StreamActivations(it),
        )
        if not ctx or not ctx.open or ctx.disabled:
            logger.error("Stream not available for nonce %s", request.nonce)
            return
        ctx.last_seq += 1
        await ctx.queue.put(
            pb2.ActivationFrame(request=request, seq=ctx.last_seq, end_of_request=False)
        )
        ctx.last_activity_t = asyncio.get_running_loop().time()

    async def _send_activation(self, msg: ActivationMessage):
        if not (self._streaming_enabled and self.next_node_stub):
            logger.error("Streaming disabled or next node not connected; cannot send")
            return
        try:
            data = self.codec.serialize(msg, self.transport_settings)
        except Exception as e:
            logger.error("Serialization failed for nonce %s: %s", msg.nonce, e)
            return
        msg.dtype = self.runtime._wire_dtype_str
        request = msg.to_proto(data)
        request.timestamp = int(time.time() * 1000)

        if self.next_node_stub is None:
            raise ValueError("next_node_stub is None")
        stub = self.next_node_stub

        ctx = await self._streams.get_or_create_stream(
            msg.nonce,
            lambda it: stub.StreamActivations(it),
        )
        if not ctx or not ctx.open or ctx.disabled:
            logger.error("Stream not available for nonce %s", msg.nonce)
            return
        ctx.last_seq += 1
        await ctx.queue.put(
            pb2.ActivationFrame(request=request, seq=ctx.last_seq, end_of_request=False)
        )
        ctx.last_activity_t = asyncio.get_running_loop().time()

        # Clear tensor reference to free memory
        try:
            msg.tensor = None
        except Exception:
            pass

    async def _send_token(self, msg: ActivationMessage) -> None:
        """
        Final-hop delivery of a sampled token to the API.

        Prefetch / offload logic lives in the compute policy; this is
        purely transport: pick an address and SendToken over gRPC.
        """
        # Pick the callback address
        cb = msg.callback_url or ""
        addr: Optional[str]

        if cb:
            parsed = urlparse(cb)
            if parsed.scheme == "grpc" and parsed.netloc:
                addr = parsed.netloc
            else:
                logger.error(
                    "Shard %s: invalid gRPC callback URL for token: %s",
                    self.runtime.shard_id,
                    cb,
                )
                return
        elif self.api_callback_address:
            # Fallback to load_model-provided address: host:port
            addr = self.api_callback_address
        else:
            logger.error(
                "Shard %s: no callback URL for final token; nonce=%s",
                self.runtime.shard_id,
                msg.nonce,
            )
            return

        try:
            if (self.api_channel is None) or (addr != self.api_address):
                # Close old channel if any
                try:
                    if self.api_channel is not None:
                        await self.api_channel.close()
                except Exception:
                    pass

                self.api_address = addr
                self.api_channel = aio_grpc.insecure_channel(
                    addr, options=GRPC_AIO_OPTIONS
                )
                self.api_stub = shard_api_comm_pb2_grpc.ShardApiServiceStub(
                    self.api_channel
                )
        except Exception as e:
            logger.error(
                "Shard %s: failed to create API channel for %s: %s",
                self.runtime.shard_id,
                addr,
                e,
            )
            return

        # send token
        t_rpc = time.perf_counter()
        try:
            token_id = int(getattr(msg, "token_id", -1))
            logprob = float(getattr(msg, "logprob", 0.0))
            top_logprobs = getattr(msg, "top_logprobs", {}) or {}

            req = shard_api_comm_pb2.TokenRequest(
                nonce=msg.nonce,
                token_id=token_id,
                timestamp=utc_epoch_now(),
                logprob=logprob,
                top_logprobs=top_logprobs,
            )

            if self.api_stub is None:
                logger.error(
                    "Shard %s: API stub not available for nonce=%s token=%s",
                    self.runtime.shard_id,
                    msg.nonce,
                    token_id,
                )
                return

            resp = await self.api_stub.SendToken(req, timeout=3.0)
            rpc_ms = (time.perf_counter() - t_rpc) * 1000.0

            if resp is None or not resp.success:
                logger.error(
                    "Shard %s: API SendToken failed for nonce=%s token=%s: %s",
                    self.runtime.shard_id,
                    msg.nonce,
                    token_id,
                    resp.message,
                )
            else:
                logger.debug(
                    "[TX-TOKEN] shard=%s nonce=%s token=%s rpc_ms=%.2f",
                    self.runtime.shard_id,
                    msg.nonce,
                    token_id,
                    rpc_ms,
                )
        except Exception as e:
            logger.exception(
                "Shard %s: error sending token via gRPC for nonce=%s: %s",
                self.runtime.shard_id,
                msg.nonce,
                e,
            )

    async def _connect_next_node(self) -> bool:
        """Connect to next node in ring.

        Returns:
            True if connected or no next node, False on failure
        """
        if not self.next_node:
            logger.info(
                "Shard %s is the final shard (no next node)", self.runtime.shard_id
            )
            return True

        if self.next_node_channel:
            logger.debug(
                "Shard %s already connected to next node.", self.runtime.shard_id
            )
            return True

        try:
            # use thunderbolt here if available
            this_properties = await self.discovery.async_get_own_properties()
            thunderbolt_conn = discover_thunderbolt_connection(
                this_properties,
                self.next_node,
            )
            next_ip = (
                thunderbolt_conn.ip_addr
                if thunderbolt_conn
                else self.next_node.local_ip
            )
            address = f"{next_ip}:{self.next_node.shard_port}"
            self.next_node_channel = aio_grpc.insecure_channel(address)
            self.next_node_stub = DnetRingServiceStub(self.next_node_channel)
            return True
        except Exception as e:
            logger.warning(
                f"Shard {self.runtime.shard_id} failed to connect to next node {address}: {e}"
            )
            self.next_node_channel = None
            self.next_node_stub = None
            return False

    async def _reconnect_next_node(self) -> bool:
        try:
            if self.next_node_channel:
                await self.next_node_channel.close()
        except Exception:
            pass
        self.next_node_channel = None
        self.next_node_stub = None
        return await self._connect_next_node()

    async def shutdown(self) -> None:
        # stop workers
        self.running = False
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            try:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            except Exception:
                pass
        self._tasks.clear()
        # close streams
        for nonce in list(self._streams._streams.keys()):
            await self._streams.end_stream(nonce)
        # close gRPC client channels
        if self.next_node_channel:
            await self.next_node_channel.close()

        self.next_node_channel = None
        self.next_node_stub = None
        if self.api_channel:
            await self.api_channel.close()
        self.api_channel = None
        self.api_stub = None

        logger.info("Shard %s: ring adapter shutdown complete", self.runtime.shard_id)
