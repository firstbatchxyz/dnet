"""
RingAdapter: ring transport + topology glue around a topology‑agnostic runtime.

- Ingress: receive ActivationRequest, stage payloads, enqueue to runtime
- Egress: read ActivationMessage from runtime and stream to next node or API
- Streaming only: no unary fallback to keep logic simple and consistent
"""
from __future__ import annotations
from typing import Optional, Any
import asyncio
import time
from urllib.parse import urlparse
import numpy as np
from dnet.ring.shard.models import ShardLoadModelRequest
import mlx.core as mx
from dnet_p2p import (
    AsyncDnetP2P,
    DnetDeviceProperties,
    discover_thunderbolt_connection,
)
from grpc import aio as aio_grpc
from .base import TopologyAdapter
from ..runtime import ShardRuntime
from .....protos.dnet_ring_pb2 import ActivationRequest
from .....protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from ....data_types import ActivationMessage
from .....utils.logger import logger
from ..stream_manager import StreamManager
from ..config import TransportConfig
from .....core.communication.activation_serializer import ActivationSerializer
from dnet.utils.serialization import dtype_map, mlx_dtype_map
from dnet.compression import decompress_tensor_from_protobuf_data
from .....protos import dnet_ring_pb2 as pb2
from .....protos import shard_api_comm_pb2, shard_api_comm_pb2_grpc

class RingAdapter(TopologyAdapter):

    def __init__(self, runtime: ShardRuntime, discovery: AsyncDnetP2P, streaming_enabled: bool,
                 transport_config: Optional[TransportConfig] = None) -> None:
        super().__init__(runtime, discovery)

        self.transport_config: TransportConfig = transport_config if transport_config else TransportConfig()

        self.running = False
        self._active_nonce: Optional[str] = None

        self._streaming_enabled = streaming_enabled
        self._streams: StreamManager = StreamManager(
            idle_timeout_s=self.transport_config.stream_idle_s,
            backoff_s=self.transport_config.stream_backoff_s,
        )

        # Topology
        self.next_node: Optional[DnetDeviceProperties] = None
        self.next_node_channel: Optional[aio_grpc.Channel] = None
        self.next_node_stub: Optional[Any] = None

        # Implement the required queues
        self._ingress_q: asyncio.Queue[ActivationRequest] = asyncio.Queue()
        self.ring_tx_q: asyncio.Queue[ActivationMessage] = asyncio.Queue()
        self.token_tx_q: asyncio.Queue[ActivationMessage] = asyncio.Queue()

        # API callback gRPC
        self.api_channel: Optional[aio_grpc.Channel] = None
        self.api_stub: Optional[Any] = None
        self.api_address: Optional[str] = None


    async def start(self):
        self.running = True
        loop = asyncio.get_running_loop()
        self._loop = loop  # if you need it in workers
        self._tasks = [
            asyncio.create_task(self._ingress_worker()),
            asyncio.create_task(self._egress_worker(loop)),
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

    async def ingress(self, data):
        pass

    async def egress(self, data):
        pass

    async def configure_topology(self, req: ShardLoadModelRequest):
        self.next_node = req.next_node
        self.total_layers = req.total_layers
        self.api_callback_address = req.api_callback_address

        if self.next_node:
            await self._connect_next_node()
        else:
            logger.warning("Node %s: No next node configured", self.runtime.shard_id)

    async def admit_frame(self, request: ActivationRequest) -> None:
        while self.running:
            try:
                self.ingress_q.put_nowait(request)
                return
            except asyncio.QueueFull:
                await asyncio.sleep(0)
        return

    async def _ingress_worker(self):
        """Drains ingress queue and processes frames with heavy work offloaded.
        """
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
                #TODO: replace it with a helper function
                if req.nonce != self._active_nonce:
                    self._active_nonce = req.nonce
                    self.runtime.get_or_make_kv(req.nonce)

                if target_layer in self.runtime._assigned_set:
                    # Heavy prep in executor (alloc/copy/decompress)
                    loop = asyncio.get_running_loop()
                    try:
                        activation_msg = await loop.run_in_executor(
                            self.runtime.executor,
                            self._prepare_activation_message_blocking,
                            req,
                        )
                    except Exception as e:
                        logger.error(
                            "Activation prepare failed for nonce %s: %s", req.nonce, e
                        )
                        continue
                    if activation_msg is None:
                        continue

                    # Enqueue for compute (cancellable back-off)
                    while self.running:
                        try:
                            self.runtime.activation_recv_queue.put_nowait(activation_msg)
                            break
                        except Exception:
                            await asyncio.sleep(0)
                    else:
                        logger.error(
                            "Failed to queue activation %s (stopping)", activation_msg.nonce
                        )
                        if self.runtime.input_pool:
                            self.runtime.input_pool.release(activation_msg.pool_id)
                else:
                    # Forward to next node (not our layer)
                    logger.debug(
                        "Forwarding activation (layer %s) to next node, nonce: %s",
                        target_layer,
                        req.nonce,
                    )
                    await self._forward_activation(req)

            except Exception as e:
                logger.error("Ingress worker error: %s", e)

    async def _egress_worker(self, loop):
        while self.running:
            msg = await loop.run_in_executor(None, self.runtime.activation_send_queue.get)
            target = self.token_tx_q if msg.is_final else self.ring_tx_q
            await target.put(msg)

    async def _ring_tx_worker(self):
        while self.running or not self.ring_tx_q.empty():
            msg = await self.ring_tx_q.get()
            await self._send_ring_activation(msg)

    async def _token_tx_worker(self):
        while self.running or not self.token_tx_q.empty():
            msg = await self.token_tx_q.get()
            await self._send_final_token(msg)

    async def _stream_sweeper(self):
        while self.running:
            await self._streams.cleanup_idle_streams()
            await asyncio.sleep(1.0)

    def _prepare_activation_message_blocking(self, request: ActivationRequest) -> Optional[ActivationMessage]:
        if self.runtime.input_pool is None:
            logger.error("Shard %s: input pool not initialized", self.runtime.shard_id)
            return None
        activation = request.activation
        if "|" in activation.dtype:
            deq = decompress_tensor_from_protobuf_data(
                tensor_data=activation.data,
                shape=list(activation.shape),
                dtype_with_metadata=activation.dtype,
            )
            pool_id = self.runtime.input_pool.allocate_for_layer(
                layer_id=activation.layer_id,
                dtype=deq.dtype,
                shape=tuple(deq.shape),
            )
            if pool_id is None:
                return None
            buffer = self.runtime.input_pool.get_buffer(pool_id)
            flat = deq.reshape(-1)
            buffer[: flat.size] = flat
            msg = ActivationMessage.from_proto(request, pool_id)
            msg.dtype = str(deq.dtype)
            msg.shape = tuple(deq.shape)
            return msg
        if activation.dtype == "tokens":
            tokens = np.frombuffer(activation.data, dtype=np.int32)
            shp = (int(len(tokens)),)
            pool_id = self.runtime.input_pool.allocate_for_layer(
                layer_id=activation.layer_id, dtype=mx.int32, shape=shp
            )
            if pool_id is None:
                return None
            buffer = self.runtime.input_pool.get_buffer(pool_id)
            buffer[: len(tokens)] = tokens
            msg = ActivationMessage.from_proto(request, pool_id)
            msg.dtype = "tokens"
            msg.shape = shp
            return msg
        expected = int(np.prod(activation.shape)) * np.dtype(dtype_map[activation.dtype]).itemsize
        actual = len(activation.data)
        if expected != actual:
            logger.error(
                "Payload size mismatch for nonce=%s: expected=%d actual=%d",
                request.nonce,
                expected,
                actual,
            )
            return None
        pool_id = self.runtime.input_pool.allocate_for_layer(
            layer_id=activation.layer_id,
            dtype=mlx_dtype_map[activation.dtype],
            shape=tuple(activation.shape),
        )
        if pool_id is None:
            return None
        buffer = self.runtime.input_pool.get_buffer(pool_id)
        input_data = np.frombuffer(activation.data, dtype=dtype_map[activation.dtype])
        buffer[: len(input_data)] = input_data
        return ActivationMessage.from_proto(request, pool_id)

    async def _forward_activation(self, request: ActivationRequest):
        if not (self._streaming_enabled and self.next_node_stub):
            logger.error("Streaming disabled or next node not connected; cannot forward")
            return
        ctx = await self._streams.get_or_create_stream(
            request.nonce, lambda it: self.next_node_stub.StreamActivations(it)  # type: ignore[attr-defined]
        )
        if not ctx or not ctx.open or ctx.disabled:
            logger.error("Stream not available for nonce %s", request.nonce)
            return
        ctx.last_seq += 1
        await ctx.queue.put(
            pb2.ActivationFrame(request=request, seq=ctx.last_seq, end_of_request=False)
        )
        ctx.last_activity_t = asyncio.get_running_loop().time()

    async def _send_ring_activation(self, msg: ActivationMessage):
        if not (self._streaming_enabled and self.next_node_stub):
            logger.error("Streaming disabled or next node not connected; cannot send")
            return
        shaped = msg.tensor
        if shaped is None:
            if self.runtime.output_pool is None:
                logger.error("No output pool and no tensor to serialize")
                return
            output_buffer = self.runtime.output_pool.get_buffer(msg.pool_id)
            data_size = int(np.prod(msg.shape))
            shaped = output_buffer[:data_size].reshape(msg.shape)

        data, _, _ = ActivationSerializer.to_bytes(
            shaped,
            wire_dtype_str=self.runtime._wire_dtype_str,
            wire_mx_dtype=self.runtime._wire_mx_dtype,
            compress=self.transport_config.compress,
            compress_min_bytes=self.transport_config.compress_min_bytes,
        )
        msg.dtype = self.runtime._wire_dtype_str
        request = msg.to_proto(data)
        request.timestamp = int(time.time() * 1000)

        ctx = await self._streams.get_or_create_stream(
            msg.nonce, lambda it: self.next_node_stub.StreamActivations(it)  # type: ignore[attr-defined]
        )
        if not ctx or not ctx.open or ctx.disabled:
            logger.error("Stream not available for nonce %s", msg.nonce)
            return
        ctx.last_seq += 1
        await ctx.queue.put(
            pb2.ActivationFrame(request=request, seq=ctx.last_seq, end_of_request=False)
        )
        ctx.last_activity_t = asyncio.get_running_loop().time()

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
