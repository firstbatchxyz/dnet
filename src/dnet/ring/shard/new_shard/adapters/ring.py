"""
RingAdapter(TopologyAdapter): everything currently ring‑specific in RingShardNode + CommsMixin:
next_node, thunderbolt discovery, _ensure_stream/_send_worker/_send_token_worker/_forward_activation, _connect_next_node.
Reads TransportConfig/TopologyConfig, owns activation_computed_queue/activation_token_queue → forwards to ring.
"""
from __future__ import annotations
from typing import Optional, Any
import asyncio
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

class RingAdapter(TopologyAdapter):

    def __init__(self, runtime: ShardRuntime, discovery: AsyncDnetP2P, streaming_enabled: bool,
                 transport_config: Optional[TransportConfig] = None) -> None:
        super().__init__(runtime, discovery)

        self.transport_config: TransportConfig = transport_config if transport_config else TransportConfig()

        self.running = False
        self._assigned_layers = runtime.shard_id.assigned_layers
        self._assigned_set = set(self._assigned_layers)

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
        self._ingress_q: asyncio.Queue[ActivationRequest]= asyncio.Queue()
        self._activation_computed_queue: asyncio.Queue[ActivationMessage] = asyncio.Queue()
        self._activation_token_queue: asyncio.Queue[ActivationMessage] = asyncio.Queue()


    async def start(self):
        self.running = True
        loop = asyncio.get_running_loop()
        self._loop = loop  # if you need it in workers
        self._tasks = [
            asyncio.create_task(self._ingress_worker()),
            asyncio.create_task(self._send_worker()),
            asyncio.create_task(self._send_token_worker()),
        ]
        if self._streaming_enabled:
            self._tasks.append(asyncio.create_task(self._stream_sweeper()))

    @property
    def ingress_q(self) -> asyncio.Queue:
        return self._ingress_q

    @property
    def activation_computed_queue(self) -> asyncio.Queue:
        return self._activation_computed_queue

    @property
    def activation_token_queue(self) -> asyncio.Queue:
        return self._activation_token_queue

    async def ingress(self, data):
        pass

    async def egress(self, data):
        pass

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

                if target_layer in self._assigned_set:
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
                        except asyncio.QueueFull:
                            await asyncio.sleep(0)
                    else:
                        logger.error("Failed to queue activation %s (node stopping)", activation_msg.nonce)
                        if self.runtime.input_pool:
                            # FIXME: !!!
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

    async def _send_worker(self):
        pass

    async def _send_token_worker(self):
        pass

    async def _connect_next_node(self) -> bool:
        """Connect to next node in ring.

        Returns:
            True if connected or no next node, False on failure
        """
        if not self.next_node:
            logger.info(f"Shard node {self.node_id} is the final shard (no next node)")
            return True

        if self.next_node_channel:
            logger.debug(f"Shard node {self.node_id} already connected to next node.")
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
                f"Shard node {self.node_id} failed to connect to next node {address}: {e}"
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