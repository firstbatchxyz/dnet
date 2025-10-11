"""Ring shard node implementation with dynamic model loading."""
import os
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue, Full
from typing import Any, Dict, List, Optional, Tuple, cast

import mlx.core as mx
import numpy as np
from fastapi import FastAPI
from grpc import aio as aio_grpc

from dnet_p2p import (
    DnetP2P,
    DnetDeviceProperties
)

from .models import (
    ShardLoadModelRequest,
    ShardLoadModelResponse,
    ShardUnloadModelResponse,
)

from ..model.base import BaseRingModel

from ...compression import decompress_tensor_from_protobuf_data
from ...protos import dnet_ring_pb2
from ...protos.shard_api_comm_pb2_grpc import ShardApiServiceStub
from ...utils.model import make_cache, load_api_layer_weights
from ...utils.logger import logger
from ...utils.layer_manager import set_prefetch_mode
from ...utils.model import ModelMetadata, get_model_metadata
from ...utils.time import utc_epoch_now
from ...utils.serialization import dtype_map, mlx_dtype_map
from ..observability import load_settings, make_profiler
from ..data_types import ActivationMessage
from ..memory_pool import LayerAwareMemoryPool
from ..model import get_ring_model
from .compute import ComputeMixin
from .prefetch import PrefetchMixin
from .send import SendMixin
from .startup import StartupMixin
from ..weight_cache import WeightCache


class RingShardNode(ComputeMixin, PrefetchMixin, SendMixin, StartupMixin):
    """Single shard node in the distributed inference ring with dynamic model loading."""

    def __init__(
        self,
        node_id: int,
        grpc_port: int,
        http_port: int,
        queue_size: int = 128,
        prefetch_threads: int = 2,
        device_prefetch_workers: int = 4,
    ) -> None:
        """Initialize ring shard node.

        Args:
            node_id: Node identifier
            grpc_port: gRPC listen port
            http_port: HTTP server port
            queue_size: Size of activation processing queue
        """
        self.node_id = node_id
        self.grpc_port = grpc_port
        self.http_port = http_port
        self.queue_size = queue_size
        self.window_size = (
            0  # Set dynamically during load_model, zero'ed for types
        )
        self._prefetch_threads = prefetch_threads
        self._device_prefetch_workers = device_prefetch_workers

        self.role: Optional[str] = None

        # Model state (loaded dynamically)
        self.model_metadata: Optional[ModelMetadata] = None
        self.assigned_layers: List[int] = []
        self.model: Optional[BaseRingModel] = None  # Ring model instance
        self.cache: Optional[Any] = None  # KV cache
        self.model_path: Optional[str] = None  # Track currently loaded model path

        from bisect import bisect_left as _bisect_left

        self._bisect_left = _bisect_left
        self._assigned_sorted = sorted(self.assigned_layers or [])
        self._assigned_set = set(self._assigned_sorted)

        # Topology (configured later)
        self.next_node: Optional[DnetDeviceProperties] = None
        self.total_layers: int = 0  # Total layers in model
        self.api_callback_address: Optional[str] = (
            None  # API callback address for final layer
        )

        # HTTP server
        self.app = FastAPI()
        self.http_server: Optional[asyncio.Task] = None

        # Memory management (initialized when model loads)
        self.input_pool: Optional[LayerAwareMemoryPool] = None
        self.output_pool: Optional[LayerAwareMemoryPool] = None
        self.weight_cache: Optional[WeightCache] = None
        self._prefetch_scheduled: set[int] = set()
        self._prefetch_pause = threading.Event()
        self._prefetch_pending: set[int] = set()
        self._prefetch_active = 0
        self._beyond_cursor: Optional[int] = None

        # Offloading params TODO: Assign these via envs or cli
        self._resident_windows = 2
        self._defer_unload = True
        self._await_next_ready = False
        set_prefetch_mode("off") # off|sequential|full
        self._warmup_keep_flag = False

        # Queues for async processing
        self.activation_recv_queue: Queue[ActivationMessage] = Queue(maxsize=queue_size)
        self.weight_prefetch_queue: Queue[int] = Queue(maxsize=50)
        self.activation_computed_queue: asyncio.Queue[ActivationMessage] = asyncio.Queue(
            maxsize=queue_size
        )
        self.ingress_q: asyncio.Queue[dnet_ring_pb2.ActivationRequest] = asyncio.Queue(maxsize=queue_size)

        # Threading
        self.compute_thread: Optional[threading.Thread] = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=int(self._device_prefetch_workers or 4))
        self._active_nonce: Optional[str] = None
        self._bound_versions: Dict[int, int] = {}
        
        try:
            _wd = (os.getenv("RING_WIRE_DTYPE", "fp16") or "fp16").strip().lower()
        except Exception:
            _wd = "fp16"
        if _wd in {"bf16", "bfloat16"}:
            self._wire_mx_dtype = mx.bfloat16
        else:
            self._wire_mx_dtype = mx.float16
        self._wire_dtype_str = str(self._wire_mx_dtype)

        # Compute serialization and MLX lock
        self._compute_busy = threading.Event()
        self._mlx_lock = threading.Lock()

        # gRPC
        self.server: Optional[aio_grpc.Server] = None
        self.next_node_channel: Optional[aio_grpc.Channel] = None
        self.next_node_stub: Optional[Any] = None
        self.api_channel: Optional[aio_grpc.Channel] = None
        self.api_stub: Optional[ShardApiServiceStub] = None
        self.api_address: Optional[str] = None

        # Discovery
        self.discovery = DnetP2P("lib/dnet-p2p/lib")

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

        obs = load_settings()
        self._profile = obs.enabled
        self._sync_per_layer = obs.sync_per_layer
        self._sync_every_n = obs.sync_every_n
        self._prof = make_profiler(self._profile)
        if self._profile:
            logger.info("[PROFILE] enabled on shard node %s", self.node_id)

        # Per-nonce KV caches (concurrent requests)
        self._kv_by_nonce: Dict[str, list] = {}
        self._kv_last_seen: Dict[str, float] = {}
        try:
            self._kv_ttl_s: float = max(1.0, float(os.getenv("RING_KV_TTL_S", "30")))
        except Exception:
            self._kv_ttl_s = 30.0

        logger.info("Shard node %s initialized with queue_size=%d", self.node_id, self.queue_size)

    async def load_model(self, req: ShardLoadModelRequest) -> ShardLoadModelResponse:
        """Load model with specified layers.
        """
        try:
            start_time = time.perf_counter()

            # Check if already loaded with same configuration
            if (
                self.model is not None
                and self.model_path == req.model_path
                and self.assigned_layers == req.layers
            ):
                logger.info("Node %s: Model already loaded with same configuration", self.node_id)
                return ShardLoadModelResponse(
                    success=True,
                    message="Model already loaded",
                    layers_loaded=req.layers,
                    load_time_ms=0.0,
                )

            # If model loaded with different config, unload first
            if self.model is not None and (
                self.model_path != req.model_path or self.assigned_layers != req.layers
            ):
                logger.info("Node %s: Unloading current model to load new configuration", self.node_id)
                await self.unload_model()

            # Load model metadata
            self.model_metadata = get_model_metadata(req.model_path)
            self.assigned_layers = req.layers
            self.model_path = req.model_path

            self.role = (
                "start"
                if 0 in self.assigned_layers
                else "end"
                if (self.model_metadata.num_layers - 1) in self.assigned_layers
                else "inter"
            )
            # Initialize memory pools
            self.input_pool = LayerAwareMemoryPool(total_memory_mb=512)
            self.output_pool = LayerAwareMemoryPool(total_memory_mb=512)

            # Set prefetch window size (must be provided by API)
            self.window_size = req.window_size
            logger.info("Node %s: Using prefetch window size: %d", self.node_id, self.window_size)

            # Initialize weight cache
            self.weight_cache = WeightCache(
                self.assigned_layers,
                self.model_metadata,
                window_size=self.window_size,
                prefetch_threads=self._prefetch_threads
            )

            # Load the model
            self.model = get_ring_model(
                self.model_metadata.model_type,
                self.model_metadata.model_config,
                assigned_layers=self.assigned_layers,
            )
            self.model.eval()
            self.cache = make_cache(self.model)

            try:
                if self.role in {"start", "end"}:
                    load_api_layer_weights(self.model_metadata, self.model)
                    logger.info("Loaded API-layer weights on shard role=%s", self.role)
            except Exception as e:
                logger.warning("Failed to load API-layer weights on role=%s: %s", self.role, e)

            # Reset prefetch tracking
            self._prefetch_scheduled = set()
            self._bound_versions = {}

            # Set topology information
            self.next_node = req.next_node
            self.total_layers = req.total_layers
            self.api_callback_address = req.api_callback_address

            if self.next_node:
                await self._connect_next_node()
            else:
                logger.info("Node %s: No next node configured", self.node_id)

            # Warmup if requested
            if req.warmup:
                await self._warmup_shard()

            #TODO: Make sure this is the right spot for prefetching
            initial_window = self._assigned_sorted[: self.window_size]
            if not (self._warmup_completed and self._warmup_keep_flag):
                for lyr in initial_window:
                    self._prefetch_to_ram(lyr)
                    self._enqueue_weight_prefetch(lyr)
            
            m = len(self._assigned_sorted)
            if m > 0:
                if m % self.window_size != 0:
                    logger.warning(
                        "Window size %s does not divide local layer count %s. Rounds per token will vary; consider setting k*w = %s.",
                        self.window_size,
                        m,
                        m,
                    )
                else:
                    k = m // self.window_size
                    logger.info("Windowed prefetch: m=%s, w=%s, k=%s rounds per token", m, self.window_size, k)

            load_time_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "Node %s: Successfully loaded model %s with layers %s in %.2fms",
                self.node_id,
                req.model_path,
                req.layers,
                load_time_ms
            )

            return ShardLoadModelResponse(
                success=True,
                message="Model loaded successfully",
                layers_loaded=req.layers,
                load_time_ms=load_time_ms,
            )

        except Exception as e:
            logger.exception("Node %s: Error loading model: %s", self.node_id, e)
            return ShardLoadModelResponse(
                success=False,
                message=f"Error loading model: {str(e)}",
                layers_loaded=[],
                load_time_ms=0.0,
            )

    async def unload_model(self) -> ShardUnloadModelResponse:
        """Unload current model and free resources.

        Returns:
            UnloadModelResponse with success and message
        """
        try:
            if self.model is None:
                return ShardUnloadModelResponse(
                    success=True,
                    message="No model loaded",
                )

            logger.info(f"Node {self.node_id}: Unloading model")

            # Clear model and cache
            self.model = None
            self.cache = None
            self.model_metadata = None
            self.assigned_layers = []
            self.model_path = None

            # Clear memory pools
            if self.weight_cache:
                # Clear all cached weights
                for layer_id in list(self._bound_versions.keys()):
                    try:
                        self.weight_cache.evict_layer(layer_id)
                    except Exception:
                        pass
                self.weight_cache = None

            self.input_pool = None
            self.output_pool = None
            self._prefetch_scheduled = set()
            self._bound_versions = {}

            # Run garbage collection to free memory
            import gc

            gc.collect()
            logger.info(f"Node {self.node_id}: Model unloaded successfully")

            return ShardUnloadModelResponse(
                success=True,
                message="Model unloaded successfully",
            )

        except Exception as e:
            logger.exception(f"Node {self.node_id}: Error unloading model: {e}")
            return ShardUnloadModelResponse(
                success=False,
                message=f"Error unloading model: {str(e)}",
            )

    def _check_model_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None and self.model_metadata is not None

    async def reset_cache(self) -> None:
        """Reset LLM KV cache."""
        if not self._check_model_loaded():
            logger.warning(f"Node {self.node_id}: Cannot reset cache - no model loaded")
            return

        try:
            self.cache = make_cache(self.model)  # type: ignore # model is checked
            logger.info(f"Node {self.node_id}: Cache reset successfully")
        except Exception as e:
            logger.error(f"Node {self.node_id}: Error resetting cache: {e}")

    async def receive_activation(self, request: dnet_ring_pb2.ActivationRequest):
        """Receive activation from previous node and queue for local compute or forward."""
        t_recv = time.perf_counter()
        await self._connect_next_node()

        try:
            activation = request.activation
            target_layer = activation.layer_id + 1

            try:
                payload_bytes = len(activation.data)
            except Exception:
                payload_bytes = -1
            transport_ms = float(utc_epoch_now() - request.timestamp)
            logger.info(
                "[PROFILE][RX] node=%s nonce=%s target_layer=%s transport_ms=%.1f payload_kb=%.1f",
                self.node_id,
                request.nonce,
                target_layer,
                transport_ms,
                (payload_bytes / 1024.0),
            )

            # Detect new sequence per node: initialize per-nonce KV; keep prefetch state per nonce
            if request.nonce != self._active_nonce:
                self._clear_prefetch_state()
                self._active_nonce = request.nonce
                try:
                    self._get_or_make_kv(request.nonce)
                except Exception:
                    pass

            if target_layer in self._assigned_set:
                # First-hop prefetch on cold start for this nonce
                if getattr(self, "_prefetch_init_nonce", None) != request.nonce:
                    next_locals = self._next_local_layers(target_layer, self.window_size - 1)
                    window_layers = [target_layer] + next_locals
                    for wl in window_layers:
                        self._prefetch_to_ram(wl)
                        self._enqueue_weight_prefetch(wl)
                    self._prefetch_init_nonce = request.nonce

                # Allocate input pool and copy payload (with optional decompression)
                t_alloc = time.perf_counter()
                if "|" in activation.dtype:
                    try:
                        deq = decompress_tensor_from_protobuf_data(
                            tensor_data=activation.data,
                            shape=list(activation.shape),
                            dtype_with_metadata=activation.dtype,
                        )
                    except Exception as e:
                        logger.error("Decompression failed for nonce %s: %s", request.nonce, e)
                        return

                    pool_id = self.input_pool.allocate_for_layer(
                        layer_id=activation.layer_id,
                        dtype=deq.dtype,
                        shape=cast(tuple[int, ...], tuple(deq.shape)),
                    )
                    if pool_id is None:
                        logger.warning("Failed to allocate input pool buffer for nonce %s", request.nonce)
                        return
                    buffer = self.input_pool.get_buffer(pool_id)
                    if buffer is not None:
                        flat = deq.reshape(-1)
                        buffer[: flat.size] = flat
                        alloc_copy_ms = (time.perf_counter() - t_alloc) * 1000.0
                        logger.info(
                            "[PROFILE][RX] node=%s nonce=%s alloc_copy_ms=%.3f (decompressed)",
                            self.node_id,
                            request.nonce,
                            alloc_copy_ms,
                        )
                    # Update activation message with true dtype/shape
                    new_dtype_str = str(deq.dtype)
                    activation_msg = ActivationMessage.from_proto(request, pool_id)
                    activation_msg.dtype = new_dtype_str
                    activation_msg.shape = tuple(deq.shape)
                else:
                    # Special token stream support: dtype='tokens' carries int32 token IDs
                    if activation.dtype == "tokens":
                        try:
                            tokens = np.frombuffer(request.activation.data, dtype=np.int32)
                            shp = (int(len(tokens)),)
                        except Exception as e:
                            logger.error("Failed to parse tokens for nonce %s: %s", request.nonce, e)
                            return
                        pool_id = self.input_pool.allocate_for_layer(
                            layer_id=activation.layer_id,
                            dtype=mx.int32,
                            shape=cast(tuple[int, ...], shp),
                        )
                        if pool_id is None:
                            logger.warning("Failed to allocate input pool buffer for nonce %s", request.nonce)
                            return
                        buffer = self.input_pool.get_buffer(pool_id)
                        if buffer is not None:
                            buffer[: len(tokens)] = tokens
                            if self._profile:
                                alloc_copy_ms = (time.perf_counter() - t_alloc) * 1000.0
                                logger.info(
                                    "[PROFILE][RX] node=%s nonce=%s alloc_copy_ms=%.3f (tokens)",
                                    self.node_id,
                                    request.nonce,
                                    alloc_copy_ms,
                                )
                        activation_msg = ActivationMessage.from_proto(request, pool_id)
                        # Ensure dtype reflects token payload for compute path
                        activation_msg.dtype = "tokens"
                        activation_msg.shape = shp
                    else:
                        # Safety: byte length must match shape*dtype
                        try:
                            expected = int(np.prod(activation.shape)) * np.dtype(dtype_map[activation.dtype]).itemsize
                            actual = len(request.activation.data)
                        except Exception:
                            expected = -1
                            actual = -1
                        if expected != actual:
                            logger.error(
                                "Payload size mismatch for nonce=%s: expected=%d actual=%d dtype=%s shape=%s",
                                request.nonce,
                                expected,
                                actual,
                                activation.dtype,
                                activation.shape,
                            )
                            return

                        pool_id = self.input_pool.allocate_for_layer(
                            layer_id=activation.layer_id,
                            dtype=mlx_dtype_map[activation.dtype],
                            shape=cast(tuple[int, ...], activation.shape),
                        )
                        if pool_id is None:
                            logger.warning("Failed to allocate input pool buffer for nonce %s", request.nonce)
                            return
                        buffer = self.input_pool.get_buffer(pool_id)
                        if buffer is not None:
                            data = request.activation.data
                            input_data = np.frombuffer(data, dtype=dtype_map[activation.dtype])
                            buffer[: len(input_data)] = input_data
                            alloc_copy_ms = (time.perf_counter() - t_alloc) * 1000.0
                            logger.info(
                                "[PROFILE][RX] node=%s nonce=%s alloc_copy_ms=%.3f",
                                self.node_id,
                                request.nonce,
                                alloc_copy_ms,
                            )
                        activation_msg = ActivationMessage.from_proto(request, pool_id)

                if self._profile:
                    activation_msg.recv_perf_t = t_recv

                # Queue for processing — non-blocking back-off loop (cancellable)
                if self._profile:
                    activation_msg.enq_perf_t = time.perf_counter()
                while self.running:
                    try:
                        self.activation_recv_queue.put_nowait(activation_msg)
                        logger.debug("Queued activation for processing: nonce %s", activation_msg.nonce)
                        break
                    except Full:
                        await asyncio.sleep(0)
                else:
                    logger.error("Failed to queue activation %s (node stopping)", activation_msg.nonce)
                    self.input_pool.release(pool_id)
            else:
                # Forward to next node (not our layer)
                logger.debug(
                    "Forwarding activation (layer %s) to next node, nonce: %s",
                    target_layer,
                    request.nonce,
                )
                await self._forward_activation(request)

        except Exception as e:
            logger.exception("Error receiving activation: %s", e)

    async def admit_frame(self, request: dnet_ring_pb2.ActivationRequest) -> None:
        """Lightweight admission for streaming: enqueue protobuf frame to ingress queue, then return.

        Uses a cancellable, non-blocking back-off loop to avoid event-loop stalls on shutdown.
        """
        while self.running:
            try:
                self.ingress_q.put_nowait(request)
                return
            except asyncio.QueueFull:
                await asyncio.sleep(0)
        # If we reached here, node is stopping; drop admission silently
        return

    async def _ingress_worker(self):
        """Drains ingress queue and processes frames with heavy work offloaded.

        Admission (servicer) is lightweight; this worker performs per-frame
        processing, offloading alloc/copy/decompress to the threadpool, and
        finally enqueues for compute or forwards to the next shard.
        """
        while self.running:
            try:
                req = await self.ingress_q.get()
            except asyncio.CancelledError:
                break
            try:
                t_recv = time.perf_counter()
                await self._connect_next_node()

                activation = req.activation
                target_layer = activation.layer_id + 1

                try:
                    payload_bytes = len(activation.data)
                except Exception:
                    payload_bytes = -1
                transport_ms = float(utc_epoch_now() - req.timestamp)
                logger.info(
                    "[PROFILE][RX] node=%s nonce=%s target_layer=%s transport_ms=%.1f payload_kb=%.1f",
                    self.node_id,
                    req.nonce,
                    target_layer,
                    transport_ms,
                    (payload_bytes / 1024.0),
                )

                # Detect new sequence per node: initialize per-nonce KV; keep prefetch state per nonce
                if req.nonce != self._active_nonce:
                    self._clear_prefetch_state()
                    self._active_nonce = req.nonce
                    try:
                        self._get_or_make_kv(req.nonce)
                    except Exception:
                        pass

                if target_layer in self._assigned_set:
                    # First-hop prefetch on cold start for this nonce
                    if getattr(self, "_prefetch_init_nonce", None) != req.nonce:
                        next_locals = self._next_local_layers(target_layer, self.window_size - 1)
                        window_layers = [target_layer] + next_locals
                        for wl in window_layers:
                            self._prefetch_to_ram(wl)
                            self._enqueue_weight_prefetch(wl)
                        self._prefetch_init_nonce = req.nonce

                    # Heavy prep in executor (alloc/copy/decompress)
                    loop = asyncio.get_running_loop()
                    try:
                        activation_msg = await loop.run_in_executor(
                            self.executor, self._prepare_activation_message_blocking, req
                        )
                    except Exception as e:
                        logger.error("Activation prepare failed for nonce %s: %s", req.nonce, e)
                        continue
                    if activation_msg is None:
                        continue
                    if self._profile:
                        activation_msg.recv_perf_t = t_recv

                    # Enqueue for compute (cancellable back-off)
                    while self.running:
                        try:
                            self.activation_recv_queue.put_nowait(activation_msg)
                            logger.debug("Queued activation for processing: nonce %s", activation_msg.nonce)
                            break
                        except Full:
                            await asyncio.sleep(0)
                    else:
                        logger.error("Failed to queue activation %s (node stopping)", activation_msg.nonce)
                        try:
                            self.input_pool.release(activation_msg.pool_id)
                        except Exception:
                            pass
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

    def _get_or_make_kv(self, nonce: str) -> list:
        """Return a per-nonce KV cache list for this shard's local layers."""
        # Opportunistic, event-driven eviction of expired nonces (no sleeps)
        try:
            now = time.perf_counter()
            ttl = float(getattr(self, "_kv_ttl_s", 30.0))
            for n, ts in list(self._kv_last_seen.items()):
                if (now - ts) > ttl:
                    self._kv_last_seen.pop(n, None)
                    self._kv_by_nonce.pop(n, None)
        except Exception:
            pass

        kv = self._kv_by_nonce.get(nonce)
        if kv is None:
            kv = make_cache(self.model)
            self._kv_by_nonce[nonce] = kv
        try:
            self._kv_last_seen[nonce] = time.perf_counter()
        except Exception:
            pass
        return kv

    def _clear_kv(self, nonce: str) -> None:
        """Clear KV for a finished nonce (no-op if missing)."""
        try:
            self._kv_by_nonce.pop(nonce, None)
            self._kv_last_seen.pop(nonce, None)
        except Exception:
            pass

    def _prepare_activation_message_blocking(
        self, request: dnet_ring_pb2.ActivationRequest
    ) -> Optional[ActivationMessage]:
        """Blocking heavy prep: allocate pool buffer, copy/decompress payload, build ActivationMessage.

        Returns None on failure.
        """
        try:
            activation = request.activation
            if "|" in activation.dtype:
                # Compressed path: decompress to MLX array and copy to pool
                try:
                    deq = decompress_tensor_from_protobuf_data(
                        tensor_data=activation.data,
                        shape=list(activation.shape),
                        dtype_with_metadata=activation.dtype,
                    )
                except Exception as e:
                    logger.error("Decompression failed for nonce %s: %s", request.nonce, e)
                    return None

                pool_id = self.input_pool.allocate_for_layer(
                    layer_id=activation.layer_id,
                    dtype=deq.dtype,
                    shape=cast(tuple[int, ...], tuple(deq.shape)),
                )
                if pool_id is None:
                    logger.warning("Failed to allocate input pool buffer for nonce %s", request.nonce)
                    return None
                buffer = self.input_pool.get_buffer(pool_id)
                if buffer is not None:
                    flat = deq.reshape(-1)
                    buffer[: flat.size] = flat
                # Update activation message with true dtype/shape
                new_dtype_str = str(deq.dtype)
                activation_msg = ActivationMessage.from_proto(request, pool_id)
                activation_msg.dtype = new_dtype_str
                activation_msg.shape = tuple(deq.shape)
                return activation_msg
            elif activation.dtype == "tokens":
                # Tokens path: parse int32 token IDs and stage them
                try:
                    tokens = np.frombuffer(activation.data, dtype=np.int32)
                    shp = (int(len(tokens)),)
                except Exception as e:
                    logger.error("Failed to parse tokens for nonce %s: %s", request.nonce, e)
                    return None
                pool_id = self.input_pool.allocate_for_layer(
                    layer_id=activation.layer_id,
                    dtype=mx.int32,
                    shape=cast(tuple[int, ...], shp),
                )
                if pool_id is None:
                    logger.warning("Failed to allocate input pool buffer for nonce %s", request.nonce)
                    return None
                buffer = self.input_pool.get_buffer(pool_id)
                if buffer is not None:
                    buffer[: len(tokens)] = tokens
                activation_msg = ActivationMessage.from_proto(request, pool_id)
                activation_msg.dtype = "tokens"
                activation_msg.shape = shp
                return activation_msg
            else:
                # Dense path: validate size and copy raw bytes view into pool buffer
                try:
                    expected = int(np.prod(activation.shape)) * np.dtype(dtype_map[activation.dtype]).itemsize
                    actual = len(activation.data)
                except Exception:
                    expected = -1
                    actual = -1
                if expected != actual:
                    logger.error(
                        "Payload size mismatch for nonce=%s: expected=%d actual=%d dtype=%s shape=%s",
                        request.nonce,
                        expected,
                        actual,
                        activation.dtype,
                        activation.shape,
                    )
                    return None

                pool_id = self.input_pool.allocate_for_layer(
                    layer_id=activation.layer_id,
                    dtype=mlx_dtype_map[activation.dtype],
                    shape=cast(tuple[int, ...], activation.shape),
                )
                if pool_id is None:
                    logger.warning("Failed to allocate input pool buffer for nonce %s", request.nonce)
                    return None
                buffer = self.input_pool.get_buffer(pool_id)
                if buffer is not None:
                    data = request.activation.data
                    input_data = np.frombuffer(data, dtype=dtype_map[activation.dtype])
                    buffer[: len(input_data)] = input_data
                activation_msg = ActivationMessage.from_proto(request, pool_id)
                return activation_msg
        except Exception as e:
            logger.error("Activation prep error: %s", e)
            return None

    def _next_local_layers(self, after_layer: int, count: int) -> List[int]:
        """Get next local layers after specified layer.

        Args:
            after_layer: Layer to start from
            count: Number of layers to return

        Returns:
            List of next local layer IDs
        """
        if count <= 0:
            return []
        ordered = sorted(self.assigned_layers)
        nxt = [lyr for lyr in ordered if lyr > after_layer]
        return nxt[:count]

    def _compute_worker(self) -> None:
        """Compute thread worker."""
        while self.running:
            try:
                # Get activation from queue (blocks until available)
                activation_msg = self.activation_recv_queue.get(timeout=1.0)

                # Process the activation
                self._process_activation(activation_msg)

            except Empty:
                continue
            except Exception as e:
                logger.error("Compute worker error: %s", e)

    async def shutdown(self) -> None:
        """Shutdown the node."""
        self.running = False

        # Stop HTTP server
        if self.http_server and not self.http_server.done():
            self.http_server.cancel()
            try:
                await self.http_server
            except asyncio.CancelledError:
                pass

        # Stop gRPC server
        if self.server:
            await self.server.stop(grace=5)

        # Close channels
        if self.next_node_channel:
            await self.next_node_channel.close()
        if self.api_channel:
            await self.api_channel.close()

        # Terminate compute thread
        if self.compute_thread:
            self.compute_thread.join(timeout=5)
        
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        # Stop discovery service
        if self.discovery.is_running():
            logger.info(f"Stopping discovery service for node {self.node_id}")
            self.discovery.stop()
            self.discovery.free_instance()
        else:
            logger.warning(f"Discovery service for node {self.node_id} was not running")

        # Stop background tasks
        for bgt in self.background_tasks:
            bgt.cancel()
        await asyncio.gather(*self.background_tasks, return_exceptions=True)

        logger.info(f"Node {self.node_id} shutdown complete")

    def _clear_prefetch_state(self):
        """Clear scheduled/in-flight prefetch when a request ends or resets"""
        try:
            self._prefetch_scheduled.clear()
        except Exception:
            pass
        try:
            self._beyond_cursor = None
        except Exception:
            pass
        try:
            while True:
                self.weight_prefetch_queue.get_nowait()
        except Exception:
            pass
        try:
            self.weight_cache.cancel_all_prefetch()
        except Exception:
            pass
