"""
ShardRuntime: owns model, KV cache, pools, windowing, weight cache, _process_activation, and the local ingress→compute→egress queues.
No ring, no gRPC, no discovery. Just: submit(ActivationIn) -> ActivationOut.
 Only knows: ingress queue in, egress queue out￼
"""

import gc
import queue
from queue import Queue
from typing import Optional, List, Any, Dict
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import mlx.core as mx

from .models import ShardLoadModelRequest, ShardUnloadModelResponse
from dnet.core.types.messages import ActivationMessage
from dnet.utils.logger import logger
from dnet.utils.model import ModelMetadata, get_model_metadata
from dnet.utils.serialization import mlx_dtype_map
from dnet.core.models import BaseRingModel as BaseShardModel, get_ring_model
import asyncio
from dnet.config import get_settings, KVCacheSettings, ComputeSettings
from dnet.core.memory.memory_pool import LayerAwareMemoryPool
from .policies import ComputePolicy, NoopPolicy, make_policy, plan_policy, PolicyPlan
from dnet.utils.model import (
    make_cache,
    load_embeddings,
    load_final_norm,
    load_lm_head,
)


# Runtime-mutable KV cache config (initialized from settings)
class RuntimeKVCacheConfig:
    """Mutable KV cache config for runtime updates."""

    def __init__(self, settings: KVCacheSettings):
        self.mode: str = settings.mode
        self.bits: int = settings.bits
        self.group_size: int = settings.group_size
        self.kv_ttl_s: float = settings.ttl_s


# Runtime-mutable compute config (initialized from settings)
class RuntimeComputeConfig:
    """Mutable compute config for runtime updates."""

    def __init__(self, settings: ComputeSettings):
        self.prefetch_mode: str = settings.prefetch_mode
        self.mxload_fastpath: bool = settings.mxload_fastpath
        self.input_pool_mb: int = settings.input_pool_mb
        self.output_pool_mb: int = settings.output_pool_mb


class ShardRuntime:
    """
    Topology-agnostic shard runtime.
    """

    def __init__(
        self,
        shard_id,
        queue_size: int = 128,
        device_prefetch_workers: int = 4,
        prefetch_threads: int = 2,
    ):
        self.shard_id = shard_id

        # Load from centralized settings
        settings = get_settings()

        # Store settings references (immutable)
        self._compute_settings = settings.compute
        self._transport_settings = settings.transport
        self._topology_settings = settings.topology

        # Mutable runtime configs (may be updated per-request or by policies)
        self.kv_cache_config = RuntimeKVCacheConfig(settings.kv_cache)
        self._compute_config = RuntimeComputeConfig(settings.compute)

        self.policy: ComputePolicy = NoopPolicy(runtime=self, resident_windows=1)

        self._device_prefetch_workers = device_prefetch_workers
        self.prefetch_threads = prefetch_threads
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Consumer queue for incoming activations/tokens
        self.max_queue_size = queue_size
        self.activation_recv_queue: Queue[ActivationMessage] = Queue(maxsize=queue_size)
        self.activation_send_queue: Queue[ActivationMessage] = Queue(maxsize=queue_size)

        self.compute_thread: Optional[threading.Thread] = None
        self.running = False
        self.executor = ThreadPoolExecutor(
            max_workers=int(self._device_prefetch_workers or 4)
        )

        # Layer assignments
        self.assigned_layers: List[int] = []
        self._assigned_sorted = sorted(self.assigned_layers or [])
        self._assigned_set = set(self._assigned_sorted)

        # Model specifics
        self.model_metadata: Optional[ModelMetadata] = None
        self.model: Optional[BaseShardModel] = None
        self.cache: Optional[Any] = None
        self.model_path: Optional[str] = None

        # Memory Pools
        self.input_pool: Optional[LayerAwareMemoryPool] = None
        self.output_pool: Optional[LayerAwareMemoryPool] = None

        # Wire dtype
        _wd = (self._transport_settings.wire_dtype or "fp16").strip().lower()
        if _wd in {"bf16", "bfloat16"}:
            self._wire_dtype_str = "bfloat16"
        else:
            self._wire_dtype_str = "float16"
        self._wire_mx_dtype = mlx_dtype_map[self._wire_dtype_str]

        # Compute serialization and MLX lock
        self._compute_busy = threading.Event()
        self._mlx_lock = threading.Lock()
        self._model_lock = threading.Lock()

        self._kv_by_nonce: Dict[str, list] = {}
        self._kv_last_seen: Dict[str, float] = {}
        self._kv_ttl_s: float = self.kv_cache_config.kv_ttl_s

    # Properties for backward compatibility
    @property
    def compute_config(self):
        """Mutable runtime compute config (for policy updates)."""
        return self._compute_config

    @property
    def transport_config(self):
        """Backward compat: returns transport settings."""
        return self._transport_settings

    @property
    def topology_config(self):
        """Backward compat: returns topology settings."""
        return self._topology_settings

    def attach_loop(self, loop):
        self._loop = loop

    def queue_size(self) -> int:
        return self.activation_recv_queue.qsize()

    def emit_result(self, msg: ActivationMessage) -> None:
        self.activation_send_queue.put_nowait(msg)

    def shutdown(self) -> None:
        # stop compute loop
        self.running = False
        if self.compute_thread:
            try:
                self.compute_thread.join(timeout=5)
            except Exception:
                pass
            self.compute_thread = None
        self.executor.shutdown(wait=True, cancel_futures=True)
        self._kv_by_nonce.clear()
        self._kv_last_seen.clear()

    def load_model_core(self, req: ShardLoadModelRequest) -> None:
        """
        load model
        """
        # Metadata + assignment
        self.model_metadata = get_model_metadata(req.model_path)
        self.assigned_layers = list(req.layers)
        self._assigned_sorted = sorted(self.assigned_layers)
        self._assigned_set = set(self._assigned_sorted)
        self.model_path = req.model_path

        local_count = max(1, len(self.assigned_layers))
        requested_w = max(1, int(req.window_size))
        n_residency = max(1, int(req.residency_size))

        plan: PolicyPlan = plan_policy(
            local_count=local_count,
            requested_w=int(req.window_size),
            residency_size=int(req.residency_size),
            topology_config=self._topology_settings,
        )

        logger.info(
            "Runtime %s: mode=%s m=%s requested_w=%s n_residency=%s -> window_size=%s resident_windows=%s is_sliding%s",
            self.shard_id,
            plan.mode,
            local_count,
            requested_w,
            n_residency,
            plan.window_size,
            plan.resident_windows,
            plan.is_sliding,
        )

        # KV cache config from API (mutable update)
        kv = (req.kv_bits or "").strip().lower()
        if kv == "4bit":
            self.kv_cache_config.mode = "4bit"
            self.kv_cache_config.bits = 4
        elif kv == "8bit":
            self.kv_cache_config.mode = "8bit"
            self.kv_cache_config.bits = 8
        else:
            # fp16 (or default)
            self.kv_cache_config.mode = "fp16"
            self.kv_cache_config.bits = max(1, self.kv_cache_config.bits)

        # Create policy + weight cache
        self.policy = make_policy(plan.mode, self, plan.resident_windows)
        self.policy.window_size = plan.window_size
        self.policy.configure_policy_for_model(req)

        self.model_metadata = get_model_metadata(self.model_path)

        # Init Pools
        self.input_pool = LayerAwareMemoryPool(
            total_memory_mb=int(self._compute_settings.input_pool_mb)
        )
        self.output_pool = LayerAwareMemoryPool(
            total_memory_mb=int(self._compute_settings.output_pool_mb)
        )

        # Load model
        self.model = get_ring_model(
            self.model_metadata.model_type,
            self.model_metadata.model_config,
            assigned_layers=self.assigned_layers,
            is_api_layer=False,
        )
        try:
            applied = bool(
                self.model.apply_quantization_from_config(
                    self.model_metadata.model_config,
                    model_metadata=self.model_metadata,
                )
            )
            logger.info(
                "[QUANT] runtime=%s applied=%s model=%s",
                self.shard_id,
                applied,
                self.model_metadata.model_type,
            )
        except RuntimeError as e:
            logger.warning("[QUANT] apply failed: %s", e)

        self.model.eval()
        self.cache = make_cache(
            self.model,
            kv_mode=self.kv_cache_config.mode,
            kv_bits=self.kv_cache_config.bits,
            kv_group=self.kv_cache_config.group_size,
        )

        # Load (embed/norm/head) if needed
        try:
            has_start = 0 in self.assigned_layers
            has_end = (self.model_metadata.num_layers - 1) in self.assigned_layers
            tied = bool(getattr(self.model.config, "tie_word_embeddings", False))
            loaded_cnt = 0
            if has_start or (has_end and tied):
                loaded_cnt += load_embeddings(self.model_metadata, self.model)
            if has_end:
                loaded_cnt += load_final_norm(self.model_metadata, self.model)
                if not tied:
                    loaded_cnt += load_lm_head(self.model_metadata, self.model)
            if loaded_cnt:
                logger.info(
                    "Runtime %s: loaded %d API‑layer tensors (start=%d end=%d tied=%d)",
                    self.shard_id,
                    loaded_cnt,
                    int(has_start),
                    int(has_end),
                    int(tied),
                )
        except Exception as e:
            logger.warning(
                "Runtime %s: failed to load API‑layer weights: %s", self.shard_id, e
            )

    def unload_model_core(self) -> ShardUnloadModelResponse:
        """
        unload model
        """
        try:
            with self._model_lock:
                if self.model is None:
                    logger.info("Node %s: No model to unload", self.shard_id)
                    return ShardUnloadModelResponse(
                        success=True,
                        message="No model loaded",
                    )

                logger.info("Node %s: Unloading model", self.shard_id)

                # Drain queue
                while not self.activation_recv_queue.empty():
                    try:
                        self.activation_recv_queue.get_nowait()
                    except queue.Empty:
                        break

                # Clear model and cache
                self.model = None
                self.cache = None
                self.model_metadata = None
                self.assigned_layers = []
                self.model_path = None
                self._assigned_sorted = []
                self._assigned_set = set()

                self.policy.clear()
                self.policy = NoopPolicy(runtime=self, resident_windows=1)
                self.input_pool = None
                self.output_pool = None

                # Run garbage collection to free memory
                gc.collect()
                mx.clear_cache()
                logger.info("Node %s: Model unloaded successfully", self.shard_id)

            return ShardUnloadModelResponse(
                success=True,
                message="Model unloaded successfully",
            )
        except Exception as e:
            logger.exception("Node %s: Error unloading model: %s", self.shard_id, e)
            return ShardUnloadModelResponse(
                success=False,
                message=f"Error unloading model: {str(e)}",
            )

    def reset_cache(self):
        if not self.model:
            logger.warning(
                "Node %s: Cannot reset cache - no model loaded", self.shard_id
            )
            return
        try:
            self.cache = make_cache(
                self.model,
                kv_mode=self.kv_cache_config.mode,
                kv_bits=self.kv_cache_config.bits,
                kv_group=self.kv_cache_config.group_size,
            )
            logger.info("Node %s: Cache reset successfully", self.shard_id)
        except Exception as e:
            logger.error("Node %s: Error resetting cache: %s", self.shard_id, e)

    def compute(self, activation_msg: ActivationMessage) -> None:
        """Replacement for _process_activation in the original shard node."""
        if not self.policy:
            logger.error("Runtime %s: no compute policy configured", self.shard_id)
            return
        self.policy.process(activation_msg)

    def _compute_worker(self) -> None:
        while self.running:
            try:
                activation_msg = self.activation_recv_queue.get(timeout=1.0)
                self.compute(activation_msg)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Compute worker error: %s", e)

    def get_or_make_kv(self, nonce: str) -> list:
        """Return a per-nonce KV cache list for this shard's local layers."""
        if not self.model:
            raise RuntimeError("Model not initialized")

        now = time.perf_counter()
        ttl = float(self._kv_ttl_s)
        for n, ts in list(self._kv_last_seen.items()):
            if (now - ts) > ttl:
                self._kv_last_seen.pop(n, None)
                self._kv_by_nonce.pop(n, None)

        kv = self._kv_by_nonce.get(nonce)
        if kv is None:
            kv = make_cache(
                self.model,
                kv_mode=self.kv_cache_config.mode,
                kv_bits=self.kv_cache_config.bits,
                kv_group=self.kv_cache_config.group_size,
            )
            self._kv_by_nonce[nonce] = kv
        self._kv_last_seen[nonce] = time.perf_counter()
        return kv

    def start(self):
        self.running = True
        self.compute_thread = threading.Thread(target=self._compute_worker, daemon=True)
        self.compute_thread.start()
