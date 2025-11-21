from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
from dnet.core.memory.weight_cache import WeightCache
from dnet.shard.models import ShardLoadModelRequest
from dnet.core.types.messages import ActivationMessage
from dnet.utils.logger import logger
import mlx.core as mx

POLICY_REGISTRY: dict[str, type["ComputePolicy"]] = {}


def register_policy(mode: str):
    def deco(cls: type[ComputePolicy]):
        POLICY_REGISTRY[mode] = cls
        return cls

    return deco


def make_policy(mode: str, runtime, resident_windows: int) -> "ComputePolicy":
    m = (mode or "fit").strip().lower()
    cls = POLICY_REGISTRY.get(m)
    if cls is None:
        raise ValueError(f"Unsupported compute mode: {m}")
    return cls(runtime, resident_windows)


# Policy is the "weights + windowing + pre/post‑compute" brain.
class ComputePolicy(ABC):
    """Abstract compute policy for ShardRuntime"""

    def __init__(self, runtime, resident_windows: int):
        self.runtime = runtime
        self.weight_cache: Optional[WeightCache] = None
        # TODO: Maybe rename this to something prefetch related?
        self._prepared_by_nonce: Dict[str, tuple[list[int], Any]] = {}

        self._resident_windows = resident_windows
        self._recent_windows: List[List[int]] = []

        self._defer_unload = True
        self._await_next_ready = False
        self._warmup_keep_flag = False
        self._warmup_completed = False

        self._bound_versions: Dict[int, int] = {}
        self._mode: Optional[str] = None
        self.window_size = 0  # set dynamically

    @abstractmethod
    def process(self, req: ActivationMessage):
        """
        Decide window layers
        waits on _prepared_by_nonce future if present
        calls weight_cache.get_weight / get_resident_layers
        does early _delta_swap_eviction (for sliding‑fit)
        """
        pass

    @abstractmethod
    def configure_policy_for_model(self, req: ShardLoadModelRequest):
        pass

    @abstractmethod
    def clear(self):
        """
        Clear any policy-specific state
        """
        pass

    @staticmethod
    def _next_local_layers(s: List[int], after_layer: int, count: int) -> List[int]:
        if count <= 0:
            return []

        for i, layer in enumerate(s):
            if layer > after_layer:
                return s[i : i + count]
        return []  # No layers found after the specified one

    def _delta_swap_eviction(
        self, window_layers: List[int], resident: List[int]
    ) -> int:
        budget = max(1, int(self.window_size or 1))
        curr_set = set(window_layers)
        prev_only = [lid for lid in resident if lid not in curr_set]
        keep_quota = max(0, budget - len(window_layers))
        idx = max(0, len(prev_only) - keep_quota)
        evict_head = prev_only[:idx]
        if not evict_head:
            return 0
        evicted: List[int] = []
        for lid in evict_head:
            try:
                if self.weight_cache and self.weight_cache.evict_layer(lid):
                    evicted.append(lid)
            except Exception:
                continue
        if evicted:
            try:
                self.runtime.model.unload_layers(evicted)
                for lid in evicted:
                    self._bound_versions.pop(lid, None)
            except Exception:
                pass
        return len(evicted)

    def _bind_layer_weights(
        self, window_layers: List[int], msg
    ) -> Optional[Dict[str, mx.array]]:
        """Bind weights for window layers if needed."""
        # Early exit if all weights are already bound and fit in window
        fast_fit = len(self.runtime._assigned_sorted) <= self.window_size
        if fast_fit and all(wl in self._bound_versions for wl in window_layers):
            return {}

        to_bind = {}
        for wl in window_layers:
            if not self.weight_cache:
                logger.error("Weight cache not initialized")
                self.runtime.input_pool.release(msg.pool_id)
                return None
            weights = self.weight_cache.get_weight(wl)
            if weights is None:
                logger.error("Failed to load weights for layer %s", wl)
                self.runtime.input_pool.release(msg.pool_id)
                return None

            # Check if weights need updating
            current_version = self._get_weight_version(weights)
            if self._bound_versions.get(wl) != current_version:
                to_bind.update(weights)
                self._bound_versions[wl] = current_version

        return to_bind

    @staticmethod
    def _get_weight_version(weights: dict) -> int:
        """Get a version identifier"""
        if not weights:
            return -1
        return id(next(iter(weights.values())))
