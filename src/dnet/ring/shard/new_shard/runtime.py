"""
ShardRuntime: owns model, KV cache, pools, windowing, weight cache, _process_activation, and the local ingress→compute→egress queues.
No ring, no gRPC, no discovery. Just: submit(ActivationIn) -> ActivationOut.
 Only knows: ingress queue in, egress queue out￼
"""
import queue
from queue import Queue
from typing import Optional, List, Any, Dict
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from ..models import ShardLoadModelRequest
from ...data_types import ActivationMessage
from ....utils.logger import logger
from ....utils.model import ModelMetadata, get_model_metadata
from ....utils.serialization import mlx_dtype_map
from ...model.base import BaseRingModel as BaseShardModel
import asyncio
from .config import ComputeConfig, TransportConfig
from ...memory_pool import LayerAwareMemoryPool
from .policies import ComputePolicy, make_policy
from ....utils.repack import ensure_repacked_for_layers
from ...model import get_ring_model
from ....utils.model import (
    make_cache,
    load_embeddings,
    load_final_norm,
    load_lm_head,
)

class ShardRuntime:
    """
    Topology-agnostic shard runtime.
    """
    def __init__(
            self,
            shard_id,
            queue_size: int = 128,
            device_prefetch_workers: int = 4,
            compute_config: Optional[ComputeConfig] = None,
            transport_config: Optional[TransportConfig] = None,
    ):

        self.shard_id = shard_id
        self.compute_config: ComputeConfig = compute_config if compute_config else ComputeConfig()
        self.transport_config: TransportConfig = transport_config if transport_config else TransportConfig()

        self.policy: Optional[ComputePolicy] = None  # ComputePolicy to be assigned later

        self._device_prefetch_workers = device_prefetch_workers
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Consumer queue for incoming activations/tokens
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
        _wd = (self.transport_config.wire_dtype or "fp16").strip().lower()
        if _wd in {"bf16", "bfloat16"}:
            self._wire_dtype_str = "bfloat16"
        else:
            self._wire_dtype_str = "float16"
        self._wire_mx_dtype = mlx_dtype_map[self._wire_dtype_str]

        # Compute serialization and MLX lock
        self._compute_busy = threading.Event()
        self._mlx_lock = threading.Lock()

        self._kv_by_nonce: Dict[str, list] = {}
        self._kv_last_seen: Dict[str, float] = {}
        self._kv_ttl_s: float = self.compute_config.kv_cache.kv_ttl_s

    def attach_loop(self, loop): self._loop = loop

    def queue_size(self) -> int: return self.activation_recv_queue.qsize()

    def emit_result(self, msg: ActivationMessage) -> None: self.activation_send_queue.put_nowait(msg)

    def start(self):
        self.running = True
        self.compute_thread = threading.Thread(target=self._compute_worker, daemon=True)
        self.compute_thread.start()

    def load_model_core(self, req: ShardLoadModelRequest) -> None:
        """
        load model
        """
        # 1) Metadata + assignment
        self.model_metadata = get_model_metadata(req.model_path)
        self.assigned_layers = list(req.layers)
        self._assigned_sorted = sorted(self.assigned_layers)
        self._assigned_set = set(self._assigned_sorted)
        self.model_path = req.model_path

        local_count = max(1, len(self.assigned_layers))
        requested_w = max(1, int(req.window_size))
        n_residency = max(1, int(req.residency_size))

        # 2) Mode + window
        if n_residency < requested_w:
            mode = "sliding_fit"
            self.compute_config.mode = mode
            resident_windows = 1  # sliding_fit preset
            window_size = max(1, min(n_residency, local_count))
        else:
            mode = "fit" if requested_w >= local_count else "offload"
            self.compute_config.mode = mode
            # let config carry resident_windows, but clamp for safety
            resident_windows = int(getattr(self.compute_config, "resident_windows", 9999))
            eff_window_size = (
                local_count
                if mode == "fit"
                else max(1, min(requested_w, local_count))
            )
            window_size = eff_window_size

        #self.policy.window_size = window_size
        #self.policy._mode = mode
        #self.policy._resident_windows = resident_windows

        logger.info(
            "Runtime %s: mode=%s m=%s requested_w=%s n_residency=%s -> window_size=%s resident_windows=%s",
            self.shard_id,
            mode,
            local_count,
            requested_w,
            n_residency,
            window_size,
            resident_windows,
        )

        # 3) KV cache config from API
        kv = (req.kv_bits or "").strip().lower()
        if kv == "4bit":
            self.compute_config.kv_cache.mode = "4bit"
            self.compute_config.kv_cache.bits = 4
        elif kv == "8bit":
            self.compute_config.kv_cache.mode = "8bit"
            self.compute_config.kv_cache.bits = 8
        else:
            # fp16 (or default)
            self.compute_config.kv_cache.mode = "fp16"
            self.compute_config.kv_cache.bits = max(
                1, int(getattr(self.compute_config.kv_cache, "bits", 8))
            )

        # 4) Repack for offload/sliding_fit (optional)
        if mode in {"sliding_fit", "offload"}:
            try:
                repacked_dir, did_repack = ensure_repacked_for_layers(
                    self.model_path, self._assigned_sorted
                )
                self.model_path = str(repacked_dir)
                self.model_metadata = get_model_metadata(self.model_path)
                self.compute_config.mxload_fastpath = True
                self.compute_config.prefetch_mode = "off"
                logger.info(
                    "[REPACK] shard=%s dst=%s layers=%s repacked=%s ms=%.1f",
                    self.shard_id,
                    self.model_path,
                    len(self._assigned_sorted),
                    int(did_repack)
                )
            except Exception as e:
                logger.warning(
                    "Runtime %s: repack failed or skipped: %s", self.shard_id, e
                )

        # 5) Pools
        self.input_pool = LayerAwareMemoryPool(
            total_memory_mb=int(self.compute_config.input_pool_mb)
        )
        self.output_pool = LayerAwareMemoryPool(
            total_memory_mb=int(self.compute_config.output_pool_mb)
        )

        # 6) Load model
        self.model = get_ring_model(
            self.model_metadata.model_type,
            self.model_metadata.model_config,
            assigned_layers=self.assigned_layers,
            is_api_layer=False,
        )
        try:
            applied = bool(
                self.model.apply_quantization_from_config(  # type: ignore[attr-defined]
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
            kv_mode=self.compute_config.kv_cache.mode,
            kv_bits=self.compute_config.kv_cache.bits,
            kv_group=self.compute_config.kv_cache.group_size,
        )

        # 7) Load API‑side weights (embed/norm/head) if needed
        try:
            has_start = 0 in self.assigned_layers
            has_end = (self.model_metadata.num_layers - 1) in self.assigned_layers
            tied = bool(
                getattr(self.model.config, "tie_word_embeddings", False)  # type: ignore[attr-defined]
            )
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
            logger.warning("Runtime %s: failed to load API‑layer weights: %s", self.shard_id, e)

        # 8) Create policy + weight cache
        self.policy = make_policy(mode, self, resident_windows)
        self.policy.window_size = window_size
        self.policy.configure_policy_for_model(req)

    def reset_cache(self):
        if not self.model:
            logger.warning(
                "Node %s: Cannot reset cache - no model loaded", self.shard_id
            )
            return
        try:
            self.cache = make_cache(
                self.model,  # type: ignore[arg-type]
                kv_mode=self.compute_config.kv_cache.mode,
                kv_bits=self.compute_config.kv_cache.bits,
                kv_group=self.compute_config.kv_cache.group_size,
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
                kv_mode=self.compute_config.kv_cache.mode,
                kv_bits=self.compute_config.kv_cache.bits,
                kv_group=self.compute_config.kv_cache.group_size,
            )
            self._kv_by_nonce[nonce] = kv
        self._kv_last_seen[nonce] = time.perf_counter()
        return kv
