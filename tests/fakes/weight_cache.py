"""Weight cache and layer manager related fakes."""

from __future__ import annotations

from typing import Any


class FakeLayerManagerForCache:
    def __init__(
        self,
        model_metadata,
        assigned_layers,
        thread_pool_size: int = 2,
        use_mxload_fastpath: bool = False,
        prefetch_mode: str = "off",
    ):
        import mlx.core as mx

        self.weight_info = getattr(model_metadata, "weight_info", {})
        self._prefetch_mode = prefetch_mode
        self._released: list[int] = []
        self._mx = mx

    def load_layer_to_gpu(self, layer_id: int):
        return {"w": self._mx.array([float(layer_id)], dtype=self._mx.float32)}

    def release_layer(self, layer_id: int) -> bool:
        self._released.append(layer_id)
        return True

    def async_prefetch(self, layer_id: int):
        from concurrent.futures import Future

        f: Future[Any] = Future()
        return f


def patch_layer_manager_for_cache(monkeypatch, cls):
    monkeypatch.setattr("dnet.core.memory.weight_cache.LayerManager", cls, raising=True)


class FakeWeightCache:
    def __init__(
        self,
        assigned_layers,
        model_metadata,
        *,
        window_size,
        prefetch_threads: int = 1,
        resident_windows: int = 1,
        use_mxload_fastpath: bool = True,
        prefetch_mode: str = "off",
    ):
        self.assigned_layers = list(assigned_layers)
        self.model_metadata = model_metadata
        self.window_size = int(window_size)
        self.prefetch_threads = int(prefetch_threads)
        self.resident_windows = int(resident_windows)
        self.use_mxload_fastpath = use_mxload_fastpath
        self.prefetch_mode = prefetch_mode
        self._weights = {}
        self.dec_refs: list[int] = []
        self.evicted: list[int] = []
        self._resident_layers: list[int] = []

    def set_resident_layers(self, layers):
        self._resident_layers = list(layers)

    def get_resident_layers(self):
        return list(self._resident_layers)

    def get_weight(self, layer_id: int):
        import mlx.core as mx

        if layer_id not in self._weights:
            self._weights[layer_id] = {
                f"w{layer_id}": mx.array([layer_id], dtype=mx.float32)
            }
        return self._weights[layer_id]

    def decrease_reference(self, layer_id: int):
        self.dec_refs.append(layer_id)

    def evict_layer(self, layer_id: int) -> bool:
        self.evicted.append(layer_id)
        return True

    # Optional helper used by OffloadPolicy offload branch tests
    def evict_layers(self, layers: list[int]) -> int:
        for lid in layers:
            self.evicted.append(lid)
        return len(layers)

    def cancel_all_prefetch(self):
        pass
