"""
Context Parallel Adapter: Implements ring attention for long-context inference.

This adapter distributes the sequence dimension across multiple devices,
with each device holding part of the context. Uses ring communication
to pass KV or Q blocks between ranks during attention computation.
"""

from __future__ import annotations

import asyncio
import queue
from typing import Optional, Callable, Awaitable, Dict
from contextvars import ContextVar
from urllib.parse import urlparse
from grpc import aio as aio_grpc

import mlx.core as mx
from dnet_p2p import AsyncDnetP2P

from dnet.core.cp.heuristics import CPAlgorithm, select_algorithm
from dnet.core.cp.ring_comm import CPRingCommunicator, RingNeighbors
from dnet.core.cp.merge_attention import (
    PartialAttentionOutput,
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

        # Operation counter for robust ring tags
        self._attn_op_counter: int = 0
        self._active_nonce: ContextVar[Optional[str]] = ContextVar(
            "active_nonce", default=None
        )
        self._current_layer_id: ContextVar[int] = ContextVar("layer_id", default=-1)
        self._current_rope_offset: ContextVar[int] = ContextVar(
            "rope_offset", default=0
        )

        # Store futures for pending ring operations
        # key: (nonce, layer_idx, step_idx) -> Future
        self._pending_ops: Dict[str, asyncio.Future] = {}

        # Persistent state for decode phase
        self._local_k_start: Optional[int] = None
        # Track prefill size per rank for decode-phase deduplication
        # During decode, non-last ranks only use prefill tokens for attention
        self._prefill_size: Optional[int] = None

    def set_active_context(self, nonce: str) -> None:
        """
        Set the active request context.
        """
        self._active_nonce.set(nonce)
        self._attn_op_counter = 0

    def reset_state(self) -> None:
        """Reset adapter state (called on cache reset)."""
        self._local_k_start = None
        self._prefill_size = None

    def set_current_layer(self, layer_id: int) -> None:
        """Set current layer ID for unique ring tags."""
        self._current_layer_id.set(layer_id)

    def set_current_rope_offset(self, offset: int) -> None:
        """Set current RoPE offset for CP calculation."""
        self._current_rope_offset.set(offset)

    @property
    def current_rope_offset(self) -> int:
        return self._current_rope_offset.get()

    @property
    def active_nonce(self) -> Optional[str]:
        return self._active_nonce.get()

    @property
    def current_layer_id(self) -> int:
        return self._current_layer_id.get()

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
        self._loop = asyncio.get_running_loop()
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

    def ring_pass_kv_attention_sync(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        rope: object = None,
        nonce: Optional[str] = None,
        layer_id: int = -1,
    ) -> mx.array:
        """
        Synchronous wrapper for ring attention, safe to call from compute threads.
        Blocks until the async ring operation on the main loop completes.
        """
        if not self.running or not hasattr(self, "_loop") or self._loop.is_closed():
            # Fallback to local if not running or loop closed
            return self._compute_attention_output(query, key, value)

        # DEBUG: Log entry to ring sync
        # logger.debug(f"CPAdapter: ring_pass_kv_attention_sync rank={self.rank_id}")

        # Safe to block because we are in ShardRuntime's compute thread, not the event loop.

        future = asyncio.run_coroutine_threadsafe(
            self.ring_pass_kv_attention(
                query, key, value, rope=rope, nonce=nonce, layer_id=layer_id
            ),
            self._loop,
        )

        try:
            return future.result()
        except Exception as e:
            logger.error(f"CPAdapter: ring_pass_kv_attention failed: {e}")
            raise

    def ring_reduce_attention_sync(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        rope: object = None,
        nonce: Optional[str] = None,
        layer_id: int = -1,
    ) -> mx.array:
        """
        Synchronous wrapper for ring reduce attention.
        """
        if not self.running or not hasattr(self, "_loop") or self._loop.is_closed():
            return self._compute_attention_output(query, key, value)

        future = asyncio.run_coroutine_threadsafe(
            self.ring_reduce_attention(
                query, key, value, rope=rope, nonce=nonce, layer_id=layer_id
            ),
            self._loop,
        )

        try:
            return future.result()
        except Exception as e:
            logger.error(f"CPAdapter: ring_reduce_attention failed: {e}")
            raise

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

        # For CP mode with multiple ranks, force load ALL layer weights before wrapping
        # This is critical because previous PP mode may have evicted/shrunk weights,
        # and the CPAttentionWrapper needs correct weights before wrapping attention modules.
        if self.num_ranks > 1 and self.runtime.model:
            logger.info(
                "CPAdapter: Forcing full weight load for %d layers before injection",
                len(self.runtime.assigned_layers),
            )
            try:
                # Get the policy's weight cache and force-load all layers
                if hasattr(self.runtime, "policy") and self.runtime.policy:
                    policy = self.runtime.policy
                    if hasattr(policy, "weight_cache") and policy.weight_cache:
                        # Force load all assigned layers and bind to model
                        all_weights = {}
                        for layer_id in self.runtime.assigned_layers:
                            w = policy.weight_cache.get_weight(layer_id, inc_ref=False)
                            if w:
                                all_weights.update(w)
                        if all_weights:
                            self.runtime.model.load_weights(
                                list(all_weights.items()), strict=False
                            )
                            logger.info(
                                "CPAdapter: Loaded %d weight tensors for CP mode",
                                len(all_weights),
                            )
            except Exception as e:
                logger.warning("CPAdapter: Failed to force-load weights: %s", e)

        # Inject ourselves into the model
        if self.runtime.model:
            logger.info("CPAdapter: Injecting logic into model")
            self.runtime.model.set_cp_adapter(self)

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

            self.ring_comm = CPRingCommunicator(
                rank_id=self.rank_id,
                num_ranks=self.num_ranks,
            )
            await self.ring_comm.connect(neighbors)

            # Access the global GrpcServer to attach our communicator
            # This is a bit hacky but we need to find the running server instance.
            # ShardRuntime -> Shard -> GrpcServer
            # But ShardRuntime doesn't know about Shard.

            # Alternative: The Shard (which owns both) should facilitate this.
            # But `configure_topology` is called via ActivationRequest... no, ShardLoadModelRequest.
            # The request comes into `ShardAdapter.configure_topology`.

            # If we can't easily reach Shard, we might need a singleton or registry.
            # OR, we verify if `runtime` has a back-reference.

            # Let's check `shard.py` to see relationships.

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

            if resp is None or not resp.success:
                logger.error(
                    "Shard %s: API SendToken failed for nonce=%s token=%s: %s",
                    self.runtime.shard_id,
                    msg.nonce,
                    token_id,
                    resp.message if resp else "no response",
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
        rope: object = None,
        send_fn: Optional[Callable[[bytes, str], Awaitable[None]]] = None,
        recv_fn: Optional[Callable[[str], Awaitable[bytes]]] = None,
        nonce: Optional[str] = None,
        layer_id: int = -1,
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

        # Query tokens are fixed in place for pass-KV.
        # Global position is provided by the absolute rope_offset.
        q_start = self.current_rope_offset

        # Local KV block starts at same position initially
        if query.shape[0] > 1:
            # Prefill: Force global offset based on rank, as runtime tracks local offset
            q_start = self.rank_id * query.shape[0]
            # Prefill: This is the start of the sequence for this shard
            self._local_k_start = q_start
            current_k_start = q_start
            # Save prefill size for decode-phase deduplication
            self._prefill_size = key.shape[0]
            # Approximate total prefill length logic for RoPE splitting later
            self._total_prefill_len = self._prefill_size * self.num_ranks

        else:
            # Decode: Use the persisted start position of the KV cache
            # q_start is the position of the NEW token, but KV cache starts at 0 (or previous start)
            if self._local_k_start is None:
                # Fallback if prefill wasn't run (unlikely but safe)
                self._local_k_start = 0
            current_k_start = self._local_k_start

        # Compute local attention first
        # Note: RoPE is already applied by CPAttentionWrapper before calling this function
        running = self._compute_partial_attention(
            query, key, value, q_start=q_start, k_start=current_k_start
        )

        current_k, current_v = key, value

        self._attn_op_counter += 1

        # Determine tag base: prefer layer ID, fallback to op counter
        tag_base = f"L{layer_id}" if layer_id >= 0 else f"op{self._attn_op_counter}"
        current_op_id = f"{nonce}_{tag_base}" if nonce else tag_base

        for step in range(1, self.num_ranks):
            # Serialize KV with its current global start position
            kv_bytes = self._serialize_kv(current_k, current_v, current_k_start)

            # Ring send/recv
            recv_bytes = await self.ring_comm.send_recv(
                kv_bytes,
                f"{current_op_id}_step{step}",
                send_fn=send_fn,
                recv_fn=recv_fn,
            )

            # Deserialize received KV and its global start position
            current_k, current_v, current_k_start = self._deserialize_kv(recv_bytes)

            # Compute attention with received KV
            # Skip if all queries are before all keys (would be fully masked by causal)
            q_end = q_start + query.shape[0] - 1  # Last query position
            k_start_pos = current_k_start  # First key position

            if q_end < k_start_pos:
                # All queries are before all keys - causal mask would block everything
                # Skip this KV block to avoid numerical issues (LSE would be -inf)
                continue

            partial = self._compute_partial_attention(
                query, current_k, current_v, q_start=q_start, k_start=current_k_start
            )

            # Online merge: accumulate into running state immediately
            running = merge_two_partials(running, partial)

        # Return merged normalized output directly
        return running.output

    async def ring_reduce_attention(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        rope: object = None,
        nonce: Optional[str] = None,
        layer_id: int = -1,
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

        # For decode: Q is the new token at position = total_kv_length
        # KV is sharded across ranks, each rank has a portion
        # Since Q is always at the END, it can attend to ALL previous tokens
        # So we skip causal masking (all positions valid)

        # DEDUPLICATION: All ranks store the same decode tokens, but we only
        # want to count them once during merge. Non-last ranks use only their
        # prefill portion for attention. Last rank uses full KV (prefill + decode).
        is_last_rank = self.rank_id == self.num_ranks - 1

        k_for_attn = key
        v_for_attn = value

        if not is_last_rank and self._prefill_size is not None:
            # Slice to prefill-only portion to avoid double-counting decode tokens
            prefill_size = self._prefill_size
            if key.shape[0] > prefill_size:
                k_for_attn = key[:prefill_size]
                v_for_attn = value[:prefill_size]

        # Compute local partial with no causal mask (decode Q > all K)
        # Note: RoPE is already applied by CPAttentionWrapper before calling this function
        running_output = self._compute_partial_attention(
            query,
            k_for_attn,
            v_for_attn,
            skip_causal_mask=True,  # Decode: Q always after K
        )

        for step in range(1, self.num_ranks):
            # Serialize current running state
            state_bytes = self._serialize_partial(running_output)

            # Ring pass
            # Tag must be unique!
            # If nonce/layer provided, use them.
            # Tag must be unique!
            # If nonce/layer provided, use them.
            tag_suffix = f"reduce_step_{step}"
            if layer_id >= 0:
                tag_suffix = f"L{layer_id}_{tag_suffix}"

            if nonce:
                tag = f"{nonce}_{tag_suffix}"
            else:
                tag = tag_suffix

            recv_bytes = await self.ring_comm.send_recv(
                state_bytes,
                tag,
            )

            # Deserialize and merge
            received_partial = self._deserialize_partial(recv_bytes)
            running_output = merge_two_partials(running_output, received_partial)

        # Return merged normalized output directly
        return running_output.output

    def _compute_partial_attention(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        q_start: int = 0,
        k_start: int = 0,
        skip_causal_mask: bool = False,
    ) -> PartialAttentionOutput:
        """
        Compute attention with tracking of max scores and log-sum-exp.

        This enables numerically stable merging of partial outputs.

        Args:
            query: Query tensor [S_q, H, D]
            key: Key tensor [S_kv, H, D]
            value: Value tensor [S_kv, H, D]
            q_start: Global starting position of query tokens (for causal mask)
            k_start: Global starting position of key tokens (for causal mask)
        """
        # Derive dimensions dynamically from tensors [S, H, D]
        S_q = query.shape[0]
        S_kv = key.shape[0]
        H_q = query.shape[1]
        H_kv = key.shape[1]
        D = query.shape[2]

        if query.shape[0] == 0:
            # Handle empty query (idle rank in CP ring)
            # Return empty tensors with correct shapes for aggregation
            return PartialAttentionOutput(
                output=mx.zeros((0, H_q, D), dtype=query.dtype),
                max_score=mx.zeros((0, H_q), dtype=query.dtype),
                log_sum_exp=mx.zeros((0, H_q), dtype=query.dtype),
            )

        # Transpose to [Heads, Seq, Dim] for correct broadcasting
        # We want to broadcast over Heads, not Sequence, because S_q != S_kv in Ring Attention
        q_h = mx.transpose(query, axes=(1, 0, 2))  # [H_q, S_q, D]
        k_h = mx.transpose(key, axes=(1, 0, 2))  # [H_kv, S_kv, D]
        v_h = mx.transpose(value, axes=(1, 0, 2))  # [H_kv, S_kv, D]

        # Handle GQA: Repeat KV heads if fewer than Q heads
        if H_kv < H_q:
            n_rep = H_q // H_kv
            if n_rep > 1:
                # k_h: [H_kv, S, D] -> [H_kv, n_rep, S, D] -> [H_q, S, D]
                k_h = mx.broadcast_to(
                    k_h[:, None],
                    (H_kv, n_rep, k_h.shape[1], k_h.shape[2]),
                )
                k_h = k_h.reshape(H_q, k_h.shape[2], k_h.shape[3])

                v_h = mx.broadcast_to(
                    v_h[:, None],
                    (H_kv, n_rep, v_h.shape[1], v_h.shape[2]),
                )
                v_h = v_h.reshape(H_q, v_h.shape[2], v_h.shape[3])

        # Scaled dot-product: QK^T / sqrt(d) -> [H, S_q, S_kv]
        scale = 1.0 / (D**0.5)
        # q_h: [H, S_q, D], k_h.T: [H, D, S_kv] -> matmul: [H, S_q, S_kv]
        scores = mx.matmul(q_h, mx.transpose(k_h, axes=(0, 2, 1))) * scale

        # Apply causal mask if needed (skip for decode where Q is always after cached K)
        if not skip_causal_mask:
            # q can only attend to k where q_global_pos >= k_global_pos
            q_positions = mx.arange(S_q) + q_start  # [S_q]
            k_positions = mx.arange(S_kv) + k_start  # [S_kv]
            # Create causal mask: [S_q, S_kv] where True = can attend
            causal_mask = q_positions[:, None] >= k_positions[None, :]  # [S_q, S_kv]
            # Apply mask: where mask is False, set score to very negative value
            # Note: -6e4 is safer than -1e9 for float16
            mask_value = mx.array(-6e4, dtype=scores.dtype)
            scores = mx.where(causal_mask, scores, mask_value)

        # Cast to float32 for softmax computation to prevent exp() overflow
        # Even with 200 tokens, attention scores can reach 35+, and exp(35) overflows float16
        original_dtype = scores.dtype
        scores_f32 = scores.astype(mx.float32)

        # Max for numerical stability
        max_score = mx.max(scores_f32, axis=-1, keepdims=False)  # [H, S_q]

        # Softmax numerator: exp(scores - max)
        exp_scores = mx.exp(scores_f32 - max_score[..., None])
        sum_exp = mx.sum(exp_scores, axis=-1, keepdims=False)  # [H, S_q]

        # NORMALIZED output: softmax @ V (standard attention output)
        attn_weights = exp_scores / sum_exp[..., None]  # Softmax in float32
        # Cast weights back to original dtype for matmul with V
        attn_weights = attn_weights.astype(original_dtype)
        # attn_weights: [H, S_q, S_kv], v_h: [H, S_kv, D] -> output: [H, S_q, D]
        output_h = mx.matmul(attn_weights, v_h)

        # Check for INF/NAN in output (Debugging)
        if mx.isnan(output_h).any() or mx.isinf(output_h).any():
            import logging

            logger = logging.getLogger("dnet")
            # Safe layer_id access
            lid = getattr(self, "current_layer_id", -1)
            logger.error(
                f"CPAdapter: INF/NAN detected in attention output! layer={lid}"
            )
            # Also check inputs to see source
            if mx.isinf(scores).any():
                logger.error("  scores has INF")
            if mx.isinf(sum_exp).any():
                logger.error("  sum_exp has INF")

        # Transpose back to [S_q, H, D]
        output = mx.transpose(output_h, axes=(1, 0, 2))

        # Transpose stats back to [S_q, H]
        max_score = mx.transpose(max_score, axes=(1, 0))
        sum_exp = mx.transpose(sum_exp, axes=(1, 0))

        # Compute proper log-sum-exp: LSE = max + log(sum_exp)
        # This is used for merging per Meta paper Eq (4)
        lse = max_score + mx.log(sum_exp + 1e-10)  # Add epsilon to avoid log(0)

        # Cast stats back to original dtype for serialization compatibility
        max_score = max_score.astype(original_dtype)
        lse = lse.astype(original_dtype)

        return PartialAttentionOutput(
            output=output,
            max_score=max_score,
            log_sum_exp=lse,  # Proper LSE for merge formula
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

    def _serialize_kv(self, key: mx.array, value: mx.array, k_start: int = 0) -> bytes:
        """Serialize KV tensors for ring transfer using Protobuf."""
        # Force evaluation of MLX arrays before serialization to ensure
        # the bytes representation is correct
        mx.eval(key)
        mx.eval(value)

        block = dnet_cp_pb2.KVBlock(
            key_data=bytes(memoryview(key)),
            value_data=bytes(memoryview(value)),
            key_shape=list(key.shape),
            value_shape=list(value.shape),
            dtype=str(key.dtype),
            k_start=k_start,
        )
        return block.SerializeToString()

    def _deserialize_kv(self, data: bytes) -> tuple[mx.array, mx.array, int]:
        """Deserialize KV tensors from bytes using Protobuf."""
        block = dnet_cp_pb2.KVBlock()
        block.ParseFromString(data)

        k = bytes_to_tensor(block.key_data, block.dtype).reshape(block.key_shape)
        v = bytes_to_tensor(block.value_data, block.dtype).reshape(block.value_shape)

        return k, v, block.k_start

    def _serialize_partial(self, partial: PartialAttentionOutput) -> bytes:
        """Serialize partial attention output for ring reduction using Protobuf."""
        # Force evaluation of MLX arrays before serialization to ensure
        # the bytes representation is correct
        mx.eval(partial.output)
        mx.eval(partial.max_score)
        mx.eval(partial.log_sum_exp)

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
