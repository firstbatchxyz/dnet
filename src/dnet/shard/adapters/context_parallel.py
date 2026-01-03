"""
Context Parallel Adapter: Implements ring attention for long-context inference.

This adapter distributes the sequence dimension across multiple devices,
with each device holding part of the context. Uses ring communication
to pass KV or Q blocks between ranks during attention computation.
"""

from __future__ import annotations

import asyncio
import queue
import time
from typing import Optional, Callable, Awaitable
from urllib.parse import urlparse
from grpc import aio as aio_grpc

import mlx.core as mx
from dnet_p2p import AsyncDnetP2P

from dnet.core.cp.heuristics import CPAlgorithm, select_algorithm
from dnet.core.cp.ring_comm import CPRingCommunicator, RingNeighbors
from dnet.core.cp.merge_attention import (
    PartialAttentionOutput,
    merge_partial_attention,
    merge_two_partials,
)
from dnet.shard.adapters.base import TopologyAdapter
from dnet.shard.runtime import ShardRuntime
from dnet.shard.models import ShardLoadModelRequest
from dnet.utils.logger import logger
from dnet.utils.grpc_config import GRPC_AIO_OPTIONS
from dnet.utils.time import utc_epoch_now
from dnet.protos.dnet_ring_pb2 import ActivationRequest
from dnet.core.types.messages import ActivationMessage
from dnet.shard.codec import ActivationCodec
from dnet.protos import shard_api_comm_pb2, shard_api_comm_pb2_grpc, dnet_cp_pb2
from dnet.utils.serialization import bytes_to_tensor


class CPAdapter(TopologyAdapter):
    """
    Context Parallel adapter for shards.

    Implements ring attention where each rank holds a portion of the sequence.
    Supports both pass-KV (prefill-optimized) and pass-Q with ring reduction
    (decode-optimized) algorithms.
    """

    def __init__(
        self,
        runtime: ShardRuntime,
        discovery: AsyncDnetP2P,
        rank_id: int = 0,
        num_ranks: int = 1,
    ):
        super().__init__(runtime, discovery)
        self.rank_id = rank_id
        self.num_ranks = num_ranks

        # Codec for activation serialization/deserialization
        self.codec = ActivationCodec(runtime)

        # Ring communicator (initialized on configure_topology)
        self.ring_comm: Optional[CPRingCommunicator] = None

        # Current algorithm selection
        self._algorithm: CPAlgorithm = CPAlgorithm.SINGLE_DEVICE

        # Model config (set on configure)
        self._num_q_heads: int = 32
        self._num_kv_heads: int = 8
        self._head_dim: int = 128

        # API callback gRPC
        self.api_channel: Optional[aio_grpc.Channel] = None
        self.api_stub: Optional[shard_api_comm_pb2_grpc.ShardApiServiceStub] = None
        self.api_address: Optional[str] = None
        self.api_callback_address: Optional[str] = None
        self._active_nonce: Optional[str] = None

        # Queues
        self.queue_size = runtime.max_queue_size
        self._ingress_q: asyncio.Queue[ActivationRequest] = asyncio.Queue(
            maxsize=self.queue_size
        )
        self._computed_q: asyncio.Queue[ActivationMessage] = asyncio.Queue(
            maxsize=self.queue_size
        )
        self._token_q: asyncio.Queue[ActivationMessage] = asyncio.Queue(
            maxsize=self.queue_size
        )

        self._tasks: list[asyncio.Task] = []

    @property
    def ingress_q(self) -> asyncio.Queue[ActivationRequest]:
        return self._ingress_q

    @property
    def activation_computed_queue(self) -> asyncio.Queue[ActivationMessage]:
        return self._computed_q

    @property
    def activation_token_queue(self) -> asyncio.Queue[ActivationMessage]:
        return self._token_q

    async def start(self) -> None:
        """Start background workers."""
        self.running = True
        self._tasks = [
            asyncio.create_task(self._ingress_worker()),
            asyncio.create_task(self._egress_worker()),
            asyncio.create_task(self._token_tx_worker()),
        ]
        logger.info(
            "CPAdapter started: rank=%d/%d, algorithm=%s",
            self.rank_id,
            self.num_ranks,
            self._algorithm,
        )

    async def ingress(self) -> None:
        """Handle incoming activation requests."""
        pass  # Handled by _ingress_worker

    async def egress(self) -> None:
        """Handle outgoing activations."""
        pass  # Handled by _egress_worker

    async def configure_topology(self, req: ShardLoadModelRequest) -> None:
        """
        Configure CP topology from load request.

        Extracts CP-specific config (rank_id, num_ranks, neighbor addresses)
        and initializes the ring communicator.
        """
        # Extract CP config using direct field access
        self.rank_id = req.cp_rank_id
        self.num_ranks = req.cp_num_ranks
        self.api_callback_address = req.api_callback_address

        # Extract model attention config for algorithm selection
        self._num_q_heads = req.num_q_heads
        self._num_kv_heads = req.num_kv_heads
        self._head_dim = req.head_dim

        # Extract neighbor addresses for ring
        rank_addresses = req.cp_rank_addresses
        if self.num_ranks > 1 and len(rank_addresses) >= self.num_ranks:
            prev_rank = (self.rank_id - 1) % self.num_ranks
            next_rank = (self.rank_id + 1) % self.num_ranks
            neighbors = RingNeighbors(
                prev_address=rank_addresses[prev_rank],
                next_address=rank_addresses[next_rank],
            )
            self.ring_comm = CPRingCommunicator(
                rank_id=self.rank_id,
                num_ranks=self.num_ranks,
            )
            await self.ring_comm.connect(neighbors)

            # CPRingServiceServicer is registered on the shard's existing gRPC server
            # (see GrpcServer.start()) - no need to start a separate server

            logger.info(
                "CPAdapter: connected ring - rank %d, prev=%s, next=%s",
                self.rank_id,
                neighbors.prev_address,
                neighbors.next_address,
            )

        else:
            self.ring_comm = CPRingCommunicator(
                rank_id=0,
                num_ranks=1,
            )

        logger.info(
            "CPAdapter configured: rank=%d/%d",
            self.rank_id,
            self.num_ranks,
        )

    async def reset_topology(self) -> None:
        """Reset topology configuration."""
        if self.ring_comm:
            await self.ring_comm.disconnect()
            self.ring_comm = None
        self.rank_id = 0
        self.num_ranks = 1

    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        self.running = False
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self.ring_comm:
            await self.ring_comm.disconnect()

        logger.info("CPAdapter: shutdown complete")

    async def _ingress_worker(self) -> None:
        """Process incoming activation requests with CP attention."""
        loop = asyncio.get_running_loop()

        while self.running:
            try:
                req = await self._ingress_q.get()
            except asyncio.CancelledError:
                break

            try:
                # Detect new nonce
                if req.nonce != self._active_nonce:
                    self._active_nonce = req.nonce
                    self.runtime.get_or_make_kv(req.nonce)

                # Deserialize and push to runtime execution queue
                activation_msg = await loop.run_in_executor(
                    self.runtime.executor,
                    self.codec.deserialize,
                    req,
                )
                if activation_msg:
                    await loop.run_in_executor(
                        None,
                        self.runtime.activation_recv_queue.put_nowait,
                        activation_msg,
                    )
            except Exception as e:
                logger.error("CPAdapter ingress error: %s", e)

    async def _egress_worker(self) -> None:
        """Forward computed activations."""
        loop = asyncio.get_running_loop()
        q = self.runtime.activation_send_queue

        while self.running:
            try:
                # Read from runtime queue
                msg = await loop.run_in_executor(
                    self.runtime.executor,
                    lambda: q.get(timeout=0.5),
                )
            except asyncio.CancelledError:
                break
            except (asyncio.QueueEmpty, queue.Empty):
                continue
            except Exception:
                continue

            # For CP, all outputs are final tokens (full replication)
            # Unless we support mixed pipeline+CP later.
            if msg.is_final:
                await self._token_q.put(msg)
            else:
                logger.warning("CPAdapter received non-final output, dropping")

    async def _token_tx_worker(self) -> None:
        """Send generated tokens back to API."""
        while self.running:
            try:
                msg = await self._token_q.get()
            except asyncio.CancelledError:
                break
            await self._send_token(msg)

    async def _send_token(self, msg: ActivationMessage) -> None:
        """
        Final-hop delivery of a sampled token to the API.
        """
        # Pick the callback address
        cb = msg.callback_url or ""
        addr: Optional[str] = None

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

    def select_algorithm_for_request(
        self,
        new_tokens: int,
        cached_tokens: int,
        batch_size: int,
    ) -> CPAlgorithm:
        """
        Select algorithm for current request based on heuristics.

        Updates self._algorithm and returns the selected algorithm.
        """
        self._algorithm = select_algorithm(
            new_tokens=new_tokens,
            cached_tokens=cached_tokens,
            batch_size=batch_size,
            num_ranks=self.num_ranks,
            num_q_heads=self._num_q_heads,
            num_kv_heads=self._num_kv_heads,
            context_parallel_enabled=(self.num_ranks > 1),
        )
        return self._algorithm

    async def ring_pass_kv_attention(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        send_fn: Optional[Callable[[bytes, str], Awaitable[None]]] = None,
        recv_fn: Optional[Callable[[str], Awaitable[bytes]]] = None,
    ) -> mx.array:
        """
        Ring attention with KV rotation (pass-KV algorithm).

        Best for full prefill where KV is smaller than Q (GQA models).

        Algorithm:
        1. Compute local attention: Attn(Q_local, KV_local)
        2. For i in 1..N-1:
            a. SendRecv: send KV to next, receive from prev
            b. Compute partial attention with received KV
            c. Accumulate partial outputs
        3. Merge all partial outputs using numerically stable merge

        Args:
            query: Local query tensor [seq_len, num_heads, head_dim]
            key: Local key tensor to rotate
            value: Local value tensor to rotate
            send_fn: Optional custom send function (for testing)
            recv_fn: Optional custom recv function (for testing)

        Returns:
            Merged attention output [seq_len, num_heads, head_dim]
        """
        if self.num_ranks == 1 or self.ring_comm is None:
            # Single device: standard attention
            return self._compute_attention_output(query, key, value)

        partials: list[PartialAttentionOutput] = []

        # Compute local attention first
        local_out = self._compute_partial_attention(query, key, value)
        partials.append(local_out)

        current_k, current_v = key, value

        for step in range(1, self.num_ranks):
            # Serialize KV for transfer
            kv_bytes = self._serialize_kv(current_k, current_v)

            # Ring send/recv: send to next, receive from prev
            recv_bytes = await self.ring_comm.send_recv(
                kv_bytes,
                f"kv_step_{step}",
                send_fn=send_fn,
                recv_fn=recv_fn,
            )

            # Deserialize received KV
            current_k, current_v = self._deserialize_kv(recv_bytes)

            # Compute attention with received KV
            partial = self._compute_partial_attention(query, current_k, current_v)
            partials.append(partial)

        # Merge all partial outputs
        return merge_partial_attention(partials)

    async def ring_reduce_attention(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
    ) -> mx.array:
        """
        Ring reduction for decode (eliminates All2All).

        Each rank computes partial attention with its local KV, then
        progressively merges partials in a ring pattern.

        Algorithm:
        1. Compute local partial = Attn(Q_all, KV_local)
        2. For step in 1..N-1:
            a. Ring pass: send running state to next, recv from prev
            b. Merge: running = merge(running, received)
        3. All ranks have fully merged output (no All2All needed!)

        Returns:
            Fully merged attention output
        """
        if self.num_ranks == 1 or self.ring_comm is None:
            return self._compute_attention_output(query, key, value)

        # Compute local partial
        running_output = self._compute_partial_attention(query, key, value)

        for step in range(1, self.num_ranks):
            # Serialize current running state
            state_bytes = self._serialize_partial(running_output)

            # Ring pass
            recv_bytes = await self.ring_comm.send_recv(
                state_bytes,
                f"reduce_step_{step}",
            )

            # Deserialize and merge
            received_partial = self._deserialize_partial(recv_bytes)
            running_output = merge_two_partials(running_output, received_partial)

        return running_output.output

    def _compute_partial_attention(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
    ) -> PartialAttentionOutput:
        """
        Compute attention with tracking of max scores and log-sum-exp.

        This enables numerically stable merging of partial outputs.
        """
        # Scaled dot-product: QK^T / sqrt(d)
        scale = 1.0 / (self._head_dim**0.5)
        scores = mx.matmul(query, mx.transpose(key, axes=(0, 2, 1))) * scale

        # Max for numerical stability
        max_score = mx.max(scores, axis=-1, keepdims=False)

        # Softmax numerator: exp(scores - max)
        exp_scores = mx.exp(scores - max_score[..., None])
        sum_exp = mx.sum(exp_scores, axis=-1, keepdims=False)

        # Attention output: softmax(scores) @ V
        attn_weights = exp_scores / sum_exp[..., None]
        output = mx.matmul(attn_weights, value)

        return PartialAttentionOutput(
            output=output,
            max_score=max_score,
            log_sum_exp=sum_exp,  # Not log yet, handled in merge
        )

    def _compute_attention_output(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
    ) -> mx.array:
        """Standard attention without partial output tracking."""
        scale = 1.0 / (self._head_dim**0.5)
        scores = mx.matmul(query, mx.transpose(key, axes=(0, 2, 1))) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        return mx.matmul(attn_weights, value)

    def _serialize_kv(self, key: mx.array, value: mx.array) -> bytes:
        """Serialize KV tensors for ring transfer using Protobuf."""
        block = dnet_cp_pb2.KVBlock(
            key_data=bytes(memoryview(key)),
            value_data=bytes(memoryview(value)),
            key_shape=list(key.shape),
            value_shape=list(value.shape),
            dtype=str(key.dtype),
        )
        return block.SerializeToString()

    def _deserialize_kv(self, data: bytes) -> tuple[mx.array, mx.array]:
        """Deserialize KV tensors from bytes using Protobuf."""
        block = dnet_cp_pb2.KVBlock()
        block.ParseFromString(data)

        k = bytes_to_tensor(block.key_data, block.dtype).reshape(block.key_shape)
        v = bytes_to_tensor(block.value_data, block.dtype).reshape(block.value_shape)

        return k, v

    def _serialize_partial(self, partial: PartialAttentionOutput) -> bytes:
        """Serialize partial attention output for ring reduction using Protobuf."""
        msg = dnet_cp_pb2.PartialOutput(
            output_data=bytes(memoryview(partial.output)),
            max_scores=bytes(memoryview(partial.max_score)),
            log_sum_exp=bytes(memoryview(partial.log_sum_exp)),
            shape=list(partial.output.shape),
            dtype=str(partial.output.dtype),
        )
        return msg.SerializeToString()

    def _deserialize_partial(self, data: bytes) -> PartialAttentionOutput:
        """Deserialize partial attention output from bytes using Protobuf."""
        msg = dnet_cp_pb2.PartialOutput()
        msg.ParseFromString(data)

        out = bytes_to_tensor(msg.output_data, msg.dtype).reshape(msg.shape)

        # Recover stats shape (B, H) from output shape (B, H, D)
        stat_shape = msg.shape[:2]
        max_s = bytes_to_tensor(msg.max_scores, msg.dtype).reshape(stat_shape)
        lse = bytes_to_tensor(msg.log_sum_exp, msg.dtype).reshape(stat_shape)

        return PartialAttentionOutput(
            output=out,
            max_score=max_s,
            log_sum_exp=lse,
        )
