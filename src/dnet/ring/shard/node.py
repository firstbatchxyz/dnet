"""Ring shard node implementation with dynamic model loading."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue, Full
from typing import Any, Dict, List, Optional, cast
from bisect import bisect_left as _bisect_left

from socket import gethostname
from secrets import token_hex

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from grpc import aio as aio_grpc

from dnet_p2p import DnetP2P, DnetDeviceProperties

from dnet.utils.latency import calculate_median_latency_seconds
from dnet.utils.serialization import tensor_to_bytes

from .servicer import ShardServicer
from dnet.protos.dnet_ring_pb2_grpc import add_DnetRingServiceServicer_to_server

from .models import (
    HealthResponse,
    ShardLoadModelRequest,
    ShardLoadModelResponse,
    ShardProfileRequest,
    ShardProfileResponse,
    ShardUnloadModelResponse,
)

from ..model.base import BaseRingModel

from ...compression import decompress_tensor_from_protobuf_data
from ...protos import dnet_ring_pb2
from ...protos.shard_api_comm_pb2_grpc import ShardApiServiceStub
from ...utils.model import (
    make_cache,
    load_embeddings,
    load_final_norm,
    load_lm_head,
)
from ...utils.logger import logger
from ...utils.layer_manager import set_prefetch_mode
from .config import ShardConfig
from ...utils.model import ModelMetadata, get_model_metadata
from ...utils.time import utc_epoch_now
from ...utils.serialization import dtype_map, mlx_dtype_map
from ..observability import load_settings, make_profiler
from ..data_types import ActivationMessage
from ..memory_pool import LayerAwareMemoryPool
from ..model import get_ring_model
from .compute import ComputeMixin
from .prefetch import PrefetchMixin
from .comms import CommsMixin
from ..weight_cache import WeightCache


class RingShardNode(ComputeMixin, PrefetchMixin, CommsMixin):
    """Single shard node in the distributed inference ring with dynamic model loading."""

    def __init__(
        self,
        node_id: int,
        grpc_port: int,
        http_port: int,
        queue_size: int = 128,
        prefetch_threads: int = 2,
        device_prefetch_workers: int = 4,
        config: Optional[ShardConfig] = None,
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
        self.window_size = 0  # Set dynamically during load_model
        self._prefetch_threads = prefetch_threads
        self._device_prefetch_workers = device_prefetch_workers

        # Model state (loaded dynamically)
        self.model_metadata: Optional[ModelMetadata] = None
        self.assigned_layers: List[int] = []
        self.model: Optional[BaseRingModel] = None  # Ring model instance
        self.cache: Optional[Any] = None  # KV cache
        self.model_path: Optional[str] = None  # Track currently loaded model path

        self._bisect_left = _bisect_left
        self._assigned_sorted = sorted(self.assigned_layers or [])
        self._assigned_set = set(self._assigned_sorted)

        # Topology (configured later)
        self.next_node: Optional[DnetDeviceProperties] = None
        self.total_layers: int = 0  # Total layers in model
        self.api_callback_address: Optional[str] = None

        # HTTP server
        self.app = FastAPI()
        self.http_server: Optional[asyncio.Task] = None

        # Configuration (preserve self._mode contract)
        self._mode = "fit"
        self.config = config or ShardConfig.for_mode(self._mode)

        # Memory management (initialized when model loads)
        self.input_pool: Optional[LayerAwareMemoryPool] = None
        self.output_pool: Optional[LayerAwareMemoryPool] = None
        self.weight_cache: Optional[WeightCache] = None
        self._prefetch_scheduled: set[int] = set()
        self._prefetch_pause = threading.Event()
        self._prefetch_pending: set[int] = set()
        self._prefetch_active = 0
        self._beyond_cursor: Optional[int] = None

        # Offloading/config-derived params
        self._resident_windows = int(self.config.resident_windows)
        self._recent_windows = []
        self._defer_unload = True
        self._await_next_ready = False
        set_prefetch_mode(self.config.prefetch_mode)
        self._warmup_keep_flag = False
        self._materialize_prefetch_default = bool(self.config.materialize_prefetch)
        self._touch_during_compute = False

        # Streaming
        self._stream_backoff_s = float(self.config.stream_backoff_s)
        self._stream_idle_s = float(self.config.stream_idle_s)
        self._send_retries = int(self.config.send_retries)
        self._explicit_eor = bool(self.config.explicit_eor)

        # Prefetch IO and touches
        self._file_io_direct = bool(self.config.file_io_direct)
        self._prefetch_touch_mode = (
            (self.config.prefetch_touch or "none").strip().lower()
        )
        self._prefetch_async = bool(self.config.prefetch_async)
        self._prefetch_fraction = float(self.config.prefetch_fraction)
        self._prefetch_budget_ms = float(self.config.prefetch_budget_ms)

        # Queues for async processing
        self.activation_recv_queue: Queue[ActivationMessage] = Queue(maxsize=queue_size)
        self.weight_prefetch_queue: Queue[int] = Queue(maxsize=50)
        self.activation_computed_queue: asyncio.Queue[ActivationMessage] = (
            asyncio.Queue(maxsize=queue_size)
        )
        self.ingress_q: asyncio.Queue[dnet_ring_pb2.ActivationRequest] = asyncio.Queue(
            maxsize=queue_size
        )

        # Threading
        self.compute_thread: Optional[threading.Thread] = None
        self.running = False
        self.executor = ThreadPoolExecutor(
            max_workers=int(self._device_prefetch_workers or 4)
        )
        self._active_nonce: Optional[str] = None
        self._bound_versions: Dict[int, int] = {}

        _wd = (self.config.wire_dtype or "fp16").strip().lower()
        if _wd in {"bf16", "bfloat16"}:
            self._wire_dtype_str = "bfloat16"
        else:
            self._wire_dtype_str = "float16"
        self._wire_mx_dtype = mlx_dtype_map[self._wire_dtype_str]

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
        self._kv_ttl_s: float = max(1.0, float(self.config.kv_cache.kv_ttl_s))

        logger.info(
            "Shard node %s initialized with queue_size=%d",
            self.node_id,
            self.queue_size,
        )

    async def load_model(self, req: ShardLoadModelRequest) -> ShardLoadModelResponse:
        """Load model with specified layers."""
        try:
            start_time = time.perf_counter()

            # Check if already loaded with same configuration
            if (
                self.model is not None
                and self.model_path == req.model_path
                and self.assigned_layers == req.layers
            ):
                logger.info(
                    "Node %s: Model already loaded with same configuration",
                    self.node_id,
                )
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
                logger.info(
                    "Node %s: Unloading current model to load new configuration",
                    self.node_id,
                )
                await self.unload_model()

            # Load model metadata
            self.model_metadata = get_model_metadata(req.model_path)
            self.assigned_layers = req.layers
            self._assigned_sorted = sorted(self.assigned_layers)
            self._assigned_set = set(self._assigned_sorted)

            self.model_path = req.model_path
            try:
                logger.info(
                    "[DIAG][LOAD] node=%s assigned_layers=%s _assigned_set_size(before)=%d",
                    self.node_id,
                    self.assigned_layers,
                    len(self._assigned_set),
                )
            except Exception:
                pass

            # Initialize memory pools
            self.input_pool = LayerAwareMemoryPool(total_memory_mb=512)
            self.output_pool = LayerAwareMemoryPool(total_memory_mb=512)

            # Decide mode dynamically from assignment + requested window
            requested_w = int(max(1, int(getattr(req, "window_size", 1))))
            local_count = max(1, len(self.assigned_layers))

            self._mode = "fit" if requested_w >= local_count else "offload"
            self.config = ShardConfig.for_mode(self._mode)  # Reset config
            eff_window_size = (
                local_count
                if (self._mode == "fit")
                else max(1, min(requested_w, local_count))
            )
            self.window_size = eff_window_size

            logger.info(
                "Node %s: Using prefetch window size: %d (requested=%s, local_count=%s, mode=%s). \n With mode: %s",
                self.node_id,
                self.window_size,
                requested_w,
                local_count,
                getattr(self.config, "mode", "fit"),
                self._mode,
            )

            # Initialize weight cache
            self.weight_cache = WeightCache(
                self.assigned_layers,
                self.model_metadata,
                window_size=self.window_size,
                prefetch_threads=self._prefetch_threads,
                resident_windows=self._resident_windows,
                file_cache_mode=self.config.file_cache_mode,
                file_dict_cap=self.config.file_dict_cap,
                eager_load=False,
            )

            # Load the model
            self.model = get_ring_model(
                self.model_metadata.model_type,
                self.model_metadata.model_config,
                assigned_layers=self.assigned_layers,
                shard_config=self.config,
            )
            self.model.eval()
            self.cache = make_cache(
                self.model,
                kv_mode=getattr(self.config.kv_cache, "mode", None),
                kv_bits=getattr(self.config.kv_cache, "bits", None),
                kv_group=getattr(self.config.kv_cache, "group_size", None),
            )

            try:
                has_start = 0 in self.assigned_layers
                has_end = (self.model_metadata.num_layers - 1) in self.assigned_layers
                tied = bool(
                    getattr(
                        getattr(self.model, "config", object()),
                        "tie_word_embeddings",
                        False,
                    )
                )

                loaded_cnt = 0
                if has_start:
                    loaded_cnt += load_embeddings(self.model_metadata, self.model)
                if has_end:
                    loaded_cnt += load_final_norm(self.model_metadata, self.model)
                    if tied:
                        # End shard needs embeddings for tied projection
                        if not has_start:
                            loaded_cnt += load_embeddings(self.model_metadata, self.model)  # fmt: skip
                        try:
                            setattr(self.model, "force_tied_head", True)
                        except Exception:
                            pass
                    else:
                        loaded_cnt += load_lm_head(self.model_metadata, self.model)
                if loaded_cnt:
                    logger.info(
                        "Loaded %d API-layer tensors (start=%d end=%d tied=%d)",
                        loaded_cnt,
                        int(has_start),
                        int(has_end),
                        int(tied),
                    )
            except Exception as e:
                logger.warning("Failed to load API-layer weights: %s", e)

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
                logger.warning("Node %s: No next node configured", self.node_id)

            # Warmup if requested (run in executor to avoid blocking event loop)
            if req.warmup:
                loop = asyncio.get_running_loop()
                try:
                    await loop.run_in_executor(self.executor, self._warmup_shard)
                except Exception:
                    # Fall back to direct call if executor is unavailable
                    self._warmup_shard()

            # TODO: Make sure this is the right spot for prefetching
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
                    logger.info(
                        "Windowed prefetch: m=%s, w=%s, k=%s rounds per token",
                        m,
                        self.window_size,
                        k,
                    )

            load_time_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "Node %s: Successfully loaded model %s with layers %s in %.2fms",
                self.node_id,
                req.model_path,
                req.layers,
                load_time_ms,
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

            logger.info("Node %s: Unloading model", self.node_id)

            # Clear model and cache
            self.model = None
            self.cache = None
            self.model_metadata = None
            self.assigned_layers = []
            self.model_path = None
            self._assigned_sorted = []
            self._assigned_set = set()

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
            mx.clear_cache()
            logger.info("Node %s: Model unloaded successfully", self.node_id)

            return ShardUnloadModelResponse(
                success=True,
                message="Model unloaded successfully",
            )

        except Exception as e:
            logger.exception("Node %s: Error unloading model: %s", self.node_id, e)
            return ShardUnloadModelResponse(
                success=False,
                message=f"Error unloading model: {str(e)}",
            )

    async def reset_cache(self) -> None:
        """Reset LLM KV cache."""
        if not self.model:
            logger.warning(
                "Node %s: Cannot reset cache - no model loaded", self.node_id
            )
            return

        try:
            self.cache = make_cache(
                self.model,  # type: ignore[arg-type]
                kv_mode=getattr(self.config.kv_cache, "mode", None),
                kv_bits=getattr(self.config.kv_cache, "bits", None),
                kv_group=getattr(self.config.kv_cache, "group_size", None),
            )
            logger.info("Node %s: Cache reset successfully", self.node_id)
        except Exception as e:
            logger.error("Node %s: Error resetting cache: %s", self.node_id, e)

    async def receive_activation(self, request: dnet_ring_pb2.ActivationRequest):
        """Receive activation from previous node and queue for local compute or forward."""
        if self.input_pool is None:
            logger.error(
                "Node %s: Cannot receive activation - input pool not initialized",
                self.node_id,
            )
            return

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
                    next_locals = self._next_local_layers(
                        target_layer, self.window_size - 1
                    )
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
                        logger.error(
                            "Decompression failed for nonce %s: %s", request.nonce, e
                        )
                        return

                    pool_id = self.input_pool.allocate_for_layer(
                        layer_id=activation.layer_id,
                        dtype=deq.dtype,
                        shape=cast(tuple[int, ...], tuple(deq.shape)),
                    )
                    if pool_id is None:
                        logger.warning(
                            "Failed to allocate input pool buffer for nonce %s",
                            request.nonce,
                        )
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
                            tokens = np.frombuffer(
                                request.activation.data, dtype=np.int32
                            )
                            shp = (int(len(tokens)),)
                        except Exception as e:
                            logger.error(
                                "Failed to parse tokens for nonce %s: %s",
                                request.nonce,
                                e,
                            )
                            return
                        pool_id = self.input_pool.allocate_for_layer(
                            layer_id=activation.layer_id,
                            dtype=mx.int32,
                            shape=cast(tuple[int, ...], shp),
                        )
                        if pool_id is None:
                            logger.warning(
                                "Failed to allocate input pool buffer for nonce %s",
                                request.nonce,
                            )
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
                            expected = (
                                int(np.prod(activation.shape))
                                * np.dtype(dtype_map[activation.dtype]).itemsize
                            )
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
                            logger.warning(
                                "Failed to allocate input pool buffer for nonce %s",
                                request.nonce,
                            )
                            return
                        buffer = self.input_pool.get_buffer(pool_id)
                        if buffer is not None:
                            data = request.activation.data
                            input_data = np.frombuffer(
                                data, dtype=dtype_map[activation.dtype]
                            )
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
                        logger.debug(
                            "Queued activation for processing: nonce %s",
                            activation_msg.nonce,
                        )
                        break
                    except Full:
                        await asyncio.sleep(0)
                else:
                    logger.error(
                        "Failed to queue activation %s (node stopping)",
                        activation_msg.nonce,
                    )
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
        """
        Lightweight admission for streaming:
        enqueue protobuf frame to ingress queue, then return.
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
                        next_locals = self._next_local_layers(
                            target_layer, self.window_size - 1
                        )
                        window_layers = [target_layer] + next_locals
                        for wl in window_layers:
                            self._prefetch_to_ram(wl)
                            self._enqueue_weight_prefetch(wl)
                        self._prefetch_init_nonce = req.nonce

                    # Heavy prep in executor (alloc/copy/decompress)
                    loop = asyncio.get_running_loop()
                    try:
                        activation_msg = await loop.run_in_executor(
                            self.executor,
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
                    if self._profile:
                        activation_msg.recv_perf_t = t_recv

                    # Enqueue for compute (cancellable back-off)
                    while self.running:
                        try:
                            self.activation_recv_queue.put_nowait(activation_msg)
                            logger.debug(
                                "Queued activation for processing: nonce %s",
                                activation_msg.nonce,
                            )
                            break
                        except Full:
                            await asyncio.sleep(0)
                    else:
                        logger.error(
                            "Failed to queue activation %s (node stopping)",
                            activation_msg.nonce,
                        )
                        try:
                            if self.input_pool:
                                # FIXME: !!!
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
        if not self.model:
            raise RuntimeError("Model not initialized")
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
            kv = make_cache(
                self.model,
                kv_mode=getattr(self.config.kv_cache, "mode", None),
                kv_bits=getattr(self.config.kv_cache, "bits", None),
                kv_group=getattr(self.config.kv_cache, "group_size", None),
            )
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
        if self.input_pool is None:
            logger.error(
                "Node %s: Cannot prepare activation - input pool not initialized",
                self.node_id,
            )
            return None

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
                    logger.error(
                        "Decompression failed for nonce %s: %s", request.nonce, e
                    )
                    return None

                pool_id = self.input_pool.allocate_for_layer(
                    layer_id=activation.layer_id,
                    dtype=deq.dtype,
                    shape=cast(tuple[int, ...], tuple(deq.shape)),
                )
                if pool_id is None:
                    logger.warning(
                        "Failed to allocate input pool buffer for nonce %s",
                        request.nonce,
                    )
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
                    logger.error(
                        "Failed to parse tokens for nonce %s: %s", request.nonce, e
                    )
                    return None
                pool_id = self.input_pool.allocate_for_layer(
                    layer_id=activation.layer_id,
                    dtype=mx.int32,
                    shape=cast(tuple[int, ...], shp),
                )
                if pool_id is None:
                    logger.warning(
                        "Failed to allocate input pool buffer for nonce %s",
                        request.nonce,
                    )
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
                    expected = (
                        int(np.prod(activation.shape))
                        * np.dtype(dtype_map[activation.dtype]).itemsize
                    )
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
                    logger.warning(
                        "Failed to allocate input pool buffer for nonce %s",
                        request.nonce,
                    )
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
        s = self._assigned_sorted
        i = _bisect_left(s, after_layer + 1)
        return s[i : i + count]

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
            if self.weight_cache:
                self.weight_cache.cancel_all_prefetch()
        except Exception:
            pass

    async def start(self, shutdown_trigger: Any = lambda: asyncio.Future()):
        self.running = True
        # Capture the main event loop for cross-thread scheduling
        try:
            self._loop = asyncio.get_running_loop()
        except Exception:
            self._loop = None
        await self._start_grpc_server()
        await self._start_http_server(shutdown_trigger)
        await asyncio.sleep(0.2)

        self.background_tasks = [
            asyncio.create_task(self._ingress_worker()),
            asyncio.create_task(self._prefetch_worker()),
            asyncio.create_task(self._send_worker()),
        ]
        # Start idle sweeper to close silent streams
        try:
            if getattr(self, "_streaming_enabled", False) and hasattr(
                self, "_stream_sweeper"
            ):
                self.background_tasks.append(
                    asyncio.create_task(self._stream_sweeper())
                )
        except Exception:
            pass

        self.compute_thread = threading.Thread(target=self._compute_worker, daemon=True)
        self.compute_thread.start()

        self._start_discovery()
        logger.info(
            "Shard node %s started on gRPC port %s HTTP port %s",
            self.node_id,
            self.grpc_port,
            self.http_port,
        )

    def _start_discovery(self) -> None:
        """Start mDNS discovery service."""

        hostname = gethostname()
        # TODO: optionally take shard name from CLI
        instance = f"shard-{token_hex(4)}-{hostname}"
        self.discovery.create_instance(
            instance,
            hostname,
            "0.0.0.0",  # Binds to all addresses
            self.http_port,  # HTTP port
            self.grpc_port,  # gRPC port
            is_manager=False,  # Shard is never a manager
        )
        self.discovery.start()
        logger.info(
            "Discovery service started for shard node %s with name %s",
            self.node_id,
            self.discovery.fullname(),
        )

    async def _start_grpc_server(self) -> None:
        """Start gRPC server."""
        self.server = aio_grpc.server()

        # Add the ring servicer; shard acts as client for ShardApiService (to API)
        servicer = ShardServicer(self)  # type: ignore # FIXME: !!!
        add_DnetRingServiceServicer_to_server(servicer, self.server)

        listen_addr = f"[::]:{self.grpc_port}"
        self.server.add_insecure_port(listen_addr)
        await self.server.start()
        logger.info(
            "Shard node %s gRPC server started on %s", self.node_id, listen_addr
        )
        try:
            await asyncio.get_running_loop().run_in_executor(
                self.executor, self._warmup_serialization
            )
            logger.info("Warmup serialization completed")
        except Exception as e:
            logger.warning("Warmup serialization failed: %s", e)

    def _warmup_serialization(self):
        try:
            dummy = mx.random.normal((1024, 1024), dtype=mx.float32)
            dummy16 = dummy.astype(self._wire_mx_dtype)
            _ = tensor_to_bytes(dummy16)
        except Exception:
            pass

    def _warmup_shard(self):
        logger.info(
            "[WARMUP] Starting shard warmup with window size %s", self.window_size
        )
        if not self.model or not self.model_metadata or not self.weight_cache:
            logger.warning("[WARMUP] No model loaded; skipping warmup")
            return

        batch_size, seq_len = 1, 1
        hidden_size = self.model_metadata.model_config.get("hidden_size", 2560)
        x = mx.zeros((batch_size, seq_len, hidden_size), dtype=mx.bfloat16)
        start_time = time.perf_counter()

        max_windows = max(1, self.config.warmup_windows)
        windows: list[list[int]] = []
        for window_start in range(0, len(self._assigned_sorted), self.window_size):
            window_end = min(
                window_start + self.window_size, len(self._assigned_sorted)
            )
            windows.append(self._assigned_sorted[window_start:window_end])
        for wi, window_layers in enumerate(windows[:max_windows]):
            weights_to_bind = {}
            for layer_id in window_layers:
                weights = self.weight_cache.get_weight(layer_id)
                if weights:
                    for k, v in weights.items():
                        weights_to_bind[k] = v
            if weights_to_bind:
                self.model.load_weights(list(weights_to_bind.items()), strict=False)
            try:
                for layer_id in window_layers:
                    x = self.model.apply_single_layer(layer_id, x, cache=None)
                    _s = mx.sum(x)
                    mx.eval(_s)
            except Exception:
                pass
            try:
                for lid in window_layers:
                    self.weight_cache.decrease_reference(lid)
            except Exception:
                pass
            if not self._warmup_keep_flag:
                try:
                    if hasattr(self.model, "unload_layers"):
                        self.model.unload_layers(window_layers)  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    self.weight_cache.evict_layers(window_layers)
                except Exception:
                    pass
        total_time = (time.perf_counter() - start_time) * 1000
        self._warmup_completed = True
        logger.info(
            "[WARMUP] Shard warmup completed in %.2fms; windows=%s kept=%s",
            total_time,
            min(len(windows), max_windows),
            int(self._warmup_keep_flag),
        )

    async def _start_http_server(self, shutdown_trigger: Any) -> None:
        """Start HTTP server.

        Args:
            shutdown_trigger: Shutdown trigger function
        """
        from hypercorn import Config
        import hypercorn.asyncio as aio_hypercorn

        await self._setup_routes()

        # Start HTTP server in background
        config = Config.from_mapping(
            bind=f"0.0.0.0:{self.http_port}",
            log_level="info",
            log_config=None,
            use_reloader=False,
            h2c=False,
        )

        # Start the server as a background task
        self.http_server = asyncio.create_task(
            aio_hypercorn.serve(self.app, config, shutdown_trigger=shutdown_trigger)  # type: ignore
        )
        logger.info(
            "Shard node %s HTTP server started on port %s", self.node_id, self.http_port
        )

    async def _setup_routes(self) -> None:
        """Setup HTTP routes."""

        @self.app.get("/health")
        async def health() -> HealthResponse:
            try:
                instance = self.discovery.instance_name()
            except Exception:
                instance = None
            return HealthResponse(
                status="ok",
                node_id=self.node_id,
                running=self.running,
                model_loaded=self.model is not None,
                model_path=self.model_path,
                assigned_layers=self.assigned_layers,
                queue_size=self.activation_recv_queue.qsize(),
                grpc_port=self.grpc_port,
                http_port=self.http_port,
                instance=instance,
            )

        @self.app.post("/profile")
        async def profile(req: ShardProfileRequest) -> ShardProfileResponse:
            logger.info("Received /profile request")
            try:
                # Measure latencies
                latency_results = await self._measure_latency_to_devices(
                    req.devices, req.thunderbolts, req.payload_sizes
                )

                # Profile device using dperf
                device_profile = await self._profile_device(
                    req.repo_id, req.max_batch_exp
                )

                # Overwrite `t_comm` with median latency (subprocess returns a dict)
                median_latency = calculate_median_latency_seconds(latency_results)
                if median_latency is not None:
                    device_profile["t_comm"] = float(median_latency)
                    logger.info(
                        f"Set t_comm to median latency: {device_profile['t_comm']:.6f}s"
                    )
                else:
                    logger.warning(
                        "No valid latency measurements, keeping default t_comm"
                    )

                # Return the dict payload directly
                return ShardProfileResponse(
                    profile=device_profile,
                    latency=latency_results,
                )
            except Exception as e:
                logger.error(f"Error in /profile endpoint: {e}")
                raise

        @self.app.post("/load_model")
        async def load_model_endpoint(
            req: ShardLoadModelRequest,
        ) -> ShardLoadModelResponse:
            """Load model with specified layers."""
            try:
                logger.info(
                    f"HTTP /load_model: model={req.model_path}, layers={req.layers}, "
                    f"next_node={req.next_node or 'none'}, window_size={req.window_size}, "
                    f"total_layers={req.total_layers}, api_callback={req.api_callback_address or 'none'}"
                )
                result = await self.load_model(req)
                return result

            except Exception as e:
                logger.error(f"Error in /load_model endpoint: {e}")
                return ShardLoadModelResponse(
                    success=False,
                    message=f"Error: {str(e)}",
                    layers_loaded=[],
                    load_time_ms=0.0,
                )

        @self.app.post("/unload_model")
        async def unload_model_endpoint() -> ShardUnloadModelResponse:
            """Unload current model."""
            try:
                logger.info("HTTP /unload_model")
                result = await self.unload_model()
                return result

            except Exception as e:
                logger.error(f"Error in /unload_model endpoint: {e}")
                return ShardUnloadModelResponse(
                    success=False,
                    message=f"Error: {str(e)}",
                )

        @self.app.post("/warm")
        # FIXME: add pydantic type here
        async def warm(request: Request) -> JSONResponse:
            try:
                body = await request.json()
                start = int(body.get("start", -1))
                window = int(body.get("window", self.window_size))
                if start < 0:
                    return JSONResponse(
                        status_code=400, content={"error": "missing/invalid start"}
                    )
                start_idx = 0
                for i, lyr in enumerate(self._assigned_sorted):
                    if lyr >= start:
                        start_idx = i
                        break
                else:
                    return JSONResponse(content={"prefetched": []})
                window_layers = self._assigned_sorted[
                    start_idx : start_idx + max(1, window)
                ]
                for wl in window_layers:
                    self._prefetch_to_ram(wl)
                    self._enqueue_weight_prefetch(wl)
                return JSONResponse(content={"prefetched": window_layers})
            except Exception as e:
                logger.error("/warm failed: %s", e)
                return JSONResponse(status_code=500, content={"error": str(e)})

    async def _profile_device(self, repo_id: str, max_batch_exp: int) -> dict:
        """Profile device using dperf in a subprocess and return a dict.

        Args:
            repo_id: Hugging Face repository ID
            max_batch_exp: Maximum batch size exponent (2^max_batch_exp)

        Returns:
            Device profile information as a plain dict
        """
        from ...utils.profile_subproc import profile_device_via_subprocess

        profile_dict = profile_device_via_subprocess(
            repo_id, max_batch_exp=max_batch_exp, debug=0
        )
        logger.info("Device profiling completed for node %s", self.node_id)
        return profile_dict

    # FIXME: this is not used, use it within healthcheck
    # this checks the health of the entire ring, but requires a bit more setup
    # e.g. it should not get into infinite loop
    async def _health_check(self):
        try:
            health_request = dnet_ring_pb2.HealthRequest(requester_id=str(self.node_id))
            response = await self.next_node_stub.HealthCheck(health_request)  # type: ignore # FIXME: this assumes an existing connection
            logger.info(
                "Shard node %s successfully pinged: %s, healthy: %s",
                self.node_id,
                response.node_id,
                response.healthy,
            )
            return True
        except Exception as e:
            logger.warning(
                "Shard node %s failed to ping next node %s",
                self.node_id,
                e,
            )
            return False
