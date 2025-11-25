"""Runtime-focused fakes used by shard/ring/policy tests."""

from __future__ import annotations

from typing import Any


class FakeRuntimeMinimal:
    """Tiny runtime with input/output pools and wire dtype defaults."""

    def __init__(self, node_id: str = "node-1"):
        self.shard_id: str = node_id
        self.assigned_layers: list[int] = []
        self.model: Any = None
        self.model_path: str | None = None
        self._queue: list = []
        self._cache_reset: bool = False
        import mlx.core as mx
        from dnet.core.memory.memory_pool import LayerAwareMemoryPool

        self._wire_dtype_str = "float16"
        self._wire_mx_dtype = mx.float16
        self.input_pool = LayerAwareMemoryPool(total_memory_mb=2)
        self.output_pool = LayerAwareMemoryPool(total_memory_mb=2)

    def attach_loop(self, loop):
        self._loop = loop

    def start(self):
        self._started = True

    def shutdown(self):
        self._started = False

    def reset_cache(self):
        self._cache_reset = True

    def queue_size(self) -> int:
        return len(self._queue)

    def load_model_core(self, req):
        self.assigned_layers = list(req.layers)
        self.model_path = req.model_path
        self.model = object()

    def unload_model_core(self):
        from dnet.shard.models import ShardUnloadModelResponse

        self.model = None
        return ShardUnloadModelResponse(success=True, message="ok")


class FakeRuntimeForAdapter:
    """Runtime stub for RingAdapter tests (queues + executor)."""

    def __init__(
        self, shard_id: str = "S1", max_queue_size: int = 8, assigned_next=set()
    ):
        import mlx.core as mx
        import queue as pyq
        from concurrent.futures import ThreadPoolExecutor

        self.shard_id = shard_id
        self.max_queue_size = max_queue_size
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.activation_recv_queue = pyq.Queue(maxsize=max_queue_size)
        self.activation_send_queue = pyq.Queue(maxsize=max_queue_size)
        self._assigned_set = set(assigned_next)
        self._wire_dtype_str = "float16"
        self._wire_mx_dtype = mx.float16
        self._kv_calls: list[str] = []

    def get_or_make_kv(self, nonce: str):
        self._kv_calls.append(nonce)
        return []

    def close(self):
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


class FakeRuntimeForPolicy:
    """Runtime stub for policy tests with pools, locks, and emit_result capture."""

    def __init__(self, assigned_layers=None, num_layers: int = 4, shard_id: str = "S1"):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import mlx.core as mx
        from dnet.core.memory.memory_pool import LayerAwareMemoryPool
        from dnet.shard.config import ComputeConfig
        from .policies import FakeComputeModel
        import types

        self.shard_id = shard_id
        self.assigned_layers = list(assigned_layers or [1, 2])
        self._assigned_sorted = sorted(self.assigned_layers)
        self._assigned_set = set(self._assigned_sorted)
        self.model_metadata = types.SimpleNamespace(num_layers=int(num_layers))
        self.model = FakeComputeModel(mx)
        self._mlx_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self.prefetch_threads = 1
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.compute_config = ComputeConfig()
        self._wire_dtype_str = "float16"
        self._wire_mx_dtype = mx.float16
        self.input_pool = LayerAwareMemoryPool(total_memory_mb=2)
        self.output_pool = LayerAwareMemoryPool(total_memory_mb=2)
        self._emitted: list = []
        self._compute_busy = threading.Event()
        self._loop = None

    def attach_loop(self, loop):
        self._loop = loop

    def get_or_make_kv(self, nonce: str):
        return []

    def emit_result(self, msg):
        self._emitted.append(msg)

    def close(self):
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
