from typing import Callable, Optional, Any
from fastapi import FastAPI
import grpc.aio as aio_grpc
import asyncio
import threading
from mlx.core import Dtype
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from dnet_p2p import DnetDeviceProperties, DnetP2P

from dnet.ring.data_types import ActivationMessage
from dnet.ring.memory_pool import LayerAwareMemoryPool
from dnet.ring.model.base import BaseRingModel
from dnet.ring.shard.config import ShardConfig
from dnet.utils.model import ModelMetadata
from dnet.ring.weight_cache import WeightCache
from dnet.ring.observability import Profiler


class RingShardNodeAttributes:
    """A mixin class that defines the attributes for a ring shard node, intended
    to be shared by all mixins & ensure type-safety among them."""

    _mlx_lock: threading.Lock

    # prefetch-related
    _prefetch_scheduled: set[int]
    _prefetch_pending: set[int]
    _prefetch_pause: threading.Event
    _prefetch_active = 0

    _streaming_enabled: bool

    _resident_windows: int
    _recent_windows: list[list[int]]
    node_id: int
    running: bool
    weight_cache: WeightCache
    weight_prefetch_queue: Queue[int]
    _materialize_prefetch_default: bool
    executor: ThreadPoolExecutor
    _touch_during_compute: bool
    _compute_busy: threading.Event

    activation_computed_queue: asyncio.Queue[ActivationMessage]
    _defer_unload: bool
    _warmup_keep_flag: bool

    # node
    grpc_port: int
    http_port: int
    app: FastAPI
    discovery: DnetP2P
    next_node: Optional[DnetDeviceProperties]
    next_node_stub: Optional[Any]

    config: ShardConfig

    _sync_per_layer: bool
    _sync_every_n: int

    # profiler
    _profile: bool
    _prof: Profiler

    input_pool: LayerAwareMemoryPool
    output_pool: Optional[LayerAwareMemoryPool]
    activation_recv_queue: Queue[ActivationMessage]

    # model
    model: Optional[BaseRingModel]
    model_metadata: Optional[ModelMetadata]
    model_path: Optional[str]

    _wire_dtype_str: str
    _wire_mx_dtype: Dtype
    _assigned_set: set[int]
    assigned_layers: list[int]
    window_size: int

    _assigned_sorted: list[int]
    _bound_versions: dict[int, int]

    next_node_channel: Optional[aio_grpc.Channel]
    next_node_stub: Optional[Any]

    # shared methods
    _prefetch_to_ram: Callable[[int], None]
    _clear_prefetch_state: Callable[[], None]
    _enqueue_weight_prefetch: Callable[[int], None]
    _next_local_layers: Callable[[int, int], list[int]]
    _get_or_make_kv: Callable[[str], list]
