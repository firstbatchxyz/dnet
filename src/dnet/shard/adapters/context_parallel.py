"""
Context Parallel Adapter: Implements ring attention for long-context inference.

This adapter distributes the sequence dimension across multiple devices,
with each device holding part of the context. Uses ring communication
to pass KV or Q blocks between ranks during attention computation.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Callable, Awaitable

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
from dnet.protos.dnet_ring_pb2 import ActivationRequest
from dnet.core.types.messages import ActivationMessage


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

        # Ring communicator (initialized on configure_topology)
        self.ring_comm: Optional[CPRingCommunicator] = None

        # Current algorithm selection
        self._algorithm: CPAlgorithm = CPAlgorithm.SINGLE_DEVICE

        # Model config (set on configure)
        self._num_q_heads: int = 32
        self._num_kv_heads: int = 8
        self._head_dim: int = 128

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
        # Extract CP config from request (will be added to ShardLoadModelRequest)
        self.rank_id = getattr(req, "cp_rank_id", 0)
        self.num_ranks = getattr(req, "cp_num_ranks", 1)

        # Extract neighbor addresses for ring
        rank_addresses = getattr(req, "cp_rank_addresses", [])
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
        while self.running:
            try:
                req = await self._ingress_q.get()
            except asyncio.CancelledError:
                break

            try:
                # TODO: Integrate with ShardRuntime for actual computation
                # For now, log and pass through
                logger.debug(
                    "CPAdapter: processing request nonce=%s, layer=%d",
                    req.nonce,
                    req.activation.layer_id,
                )
            except Exception as e:
                logger.error("CPAdapter ingress error: %s", e)

    async def _egress_worker(self) -> None:
        """Forward computed activations."""
        while self.running:
            try:
                msg = await self._computed_q.get()
            except asyncio.CancelledError:
                break

            # Forward to token queue if final, else to ring
            if msg.is_final:
                await self._token_q.put(msg)

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
        """Serialize KV tensors for ring transfer."""
        # Use memoryview for mx.array serialization
        k_bytes = bytes(memoryview(key))
        v_bytes = bytes(memoryview(value))
        # Pack: k_len (4 bytes) + k_bytes + v_bytes
        k_len = len(k_bytes).to_bytes(4, "little")
        return k_len + k_bytes + v_bytes

    def _deserialize_kv(self, data: bytes) -> tuple[mx.array, mx.array]:
        """Deserialize KV tensors from bytes."""
        k_len = int.from_bytes(data[:4], "little")
        _k_bytes = data[4 : 4 + k_len]  # noqa: F841 - placeholder
        _v_bytes = data[4 + k_len :]  # noqa: F841 - placeholder
        # TODO: Need shape info to reconstruct properly
        # For now, return empty arrays as placeholder
        return mx.zeros((1,)), mx.zeros((1,))

    def _serialize_partial(self, partial: PartialAttentionOutput) -> bytes:
        """Serialize partial attention output for ring reduction."""
        out_bytes = bytes(memoryview(partial.output))
        max_bytes = bytes(memoryview(partial.max_score))
        lse_bytes = bytes(memoryview(partial.log_sum_exp))
        # Pack lengths
        out_len = len(out_bytes).to_bytes(4, "little")
        max_len = len(max_bytes).to_bytes(4, "little")
        return out_len + max_len + out_bytes + max_bytes + lse_bytes

    def _deserialize_partial(self, data: bytes) -> PartialAttentionOutput:
        """Deserialize partial attention output from bytes."""
        _out_len = int.from_bytes(data[:4], "little")  # noqa: F841 - placeholder
        _max_len = int.from_bytes(data[4:8], "little")  # noqa: F841 - placeholder
        # TODO: Need shape info to reconstruct properly
        return PartialAttentionOutput(
            output=mx.zeros((1,)),
            max_score=mx.zeros((1,)),
            log_sum_exp=mx.zeros((1,)),
        )
