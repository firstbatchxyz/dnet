"""Tests: WeightCache capacity, hit/miss, eviction, ref-counts, and release."""

import threading
import pytest
import time
from concurrent.futures import Future


mx = pytest.importorskip("mlx.core")

from dnet.core.memory.weight_cache import WeightCache
from tests.fakes import (
    FakeLayerManagerForCache,
    FakeModelMetadata,
    FakeWeightSize,
    patch_layer_manager_for_cache,
)

pytestmark = [pytest.mark.core]


def test_init_resident_budget(monkeypatch):
    patch_layer_manager_for_cache(monkeypatch, FakeLayerManagerForCache)
    meta = FakeModelMetadata({})
    wc = WeightCache([0, 1, 2], meta, window_size=1, resident_windows=2)
    assert wc.max_weights == 2
    wc2 = WeightCache([0, 1, 2], meta, window_size=None)
    assert wc2.max_weights == 3


def test_get_weight_cache_hit(monkeypatch):
    patch_layer_manager_for_cache(monkeypatch, FakeLayerManagerForCache)
    meta = FakeModelMetadata({})
    wc = WeightCache([5], meta, window_size=1)
    data = {"w": mx.array([1.0])}
    wc.cache[5] = (data, 0.0)
    wc.reference_counts[5] = 0
    out = wc.get_weight(5, inc_ref=True)
    assert out is data
    assert wc.reference_counts[5] == 1
    assert wc.cache[5][1] > 0.0


def test_get_weight_cache_hit_no_inc_ref(monkeypatch):
    patch_layer_manager_for_cache(monkeypatch, FakeLayerManagerForCache)
    meta = FakeModelMetadata({})
    wc = WeightCache([6], meta, window_size=1)
    data = {"w": mx.array([1.0])}
    wc.cache[6] = (data, 0.0)
    wc.reference_counts[6] = 0
    out = wc.get_weight(6, inc_ref=False)
    assert out is data
    assert wc.reference_counts[6] == 0


def test_get_weight_creator_success(monkeypatch):
    patch_layer_manager_for_cache(monkeypatch, FakeLayerManagerForCache)
    meta = FakeModelMetadata({7: {"w": FakeWeightSize(4)}})
    wc = WeightCache([7], meta, window_size=1)
    out = wc.get_weight(7)
    assert out and "w" in out
    assert 7 in wc.cache and wc.reference_counts.get(7, 0) == 1
    assert 7 not in wc.loading_futures


def test_get_weight_inflight_wait(monkeypatch):
    patch_layer_manager_for_cache(monkeypatch, FakeLayerManagerForCache)
    meta = FakeModelMetadata({8: {"w": FakeWeightSize(4)}})
    wc = WeightCache([8], meta, window_size=1)
    f = Future()
    with wc.lock:
        wc.loading_futures[8] = f

    def fulfill():
        time.sleep(0.01)
        with wc.lock:
            wc.cache[8] = ({"w": mx.array([8.0])}, time.time())
        f.set_result(True)

    t = threading.Thread(target=fulfill)
    t.start()
    out = wc.get_weight(8)
    t.join()
    assert out and "w" in out and float(out["w"][0]) == 8.0
    assert wc.reference_counts.get(8, 0) == 1


def test_get_weight_inflight_exception_returns_none(monkeypatch):
    patch_layer_manager_for_cache(monkeypatch, FakeLayerManagerForCache)
    meta = FakeModelMetadata({10: {"w": FakeWeightSize(4)}})
    wc = WeightCache([10], meta, window_size=1)
    f = Future()
    f.set_exception(RuntimeError("boom"))
    with wc.lock:
        wc.loading_futures[10] = f
    out = wc.get_weight(10)
    assert out is None


def test_get_weight_creator_failure(monkeypatch):
    class _FailLM(FakeLayerManagerForCache):
        def load_layer_to_gpu(self, layer_id):
            raise RuntimeError("boom")

    patch_layer_manager_for_cache(monkeypatch, _FailLM)
    meta = FakeModelMetadata({9: {"w": FakeWeightSize(4)}})
    wc = WeightCache([9], meta, window_size=1)
    out = wc.get_weight(9)
    assert out is None
    assert 9 not in wc.loading_futures


def test_evict_lru_calls_release(monkeypatch):
    lm = FakeLayerManagerForCache(FakeModelMetadata({}), [])
    patch_layer_manager_for_cache(monkeypatch, lambda *a, **k: lm)
    wc = WeightCache([0, 1], FakeModelMetadata({}), window_size=1)
    wc.cache[0] = ({"w": mx.array([0.0])}, 1.0)
    wc.cache[1] = ({"w": mx.array([1.0])}, 2.0)
    wc.reference_counts = {0: 0, 1: 0}
    wc._evict_lru()
    assert 0 not in wc.cache and 0 in lm._released


def test_decrease_reference_allows_eviction(monkeypatch):
    lm = FakeLayerManagerForCache(FakeModelMetadata({}), [])
    patch_layer_manager_for_cache(monkeypatch, lambda *a, **k: lm)
    wc = WeightCache([1], FakeModelMetadata({}), window_size=1)
    wc.cache[1] = ({"w": mx.array([1.0])}, 0.0)
    wc.reference_counts[1] = 1
    wc.decrease_reference(1)
    assert wc.evict_layer(1) is True


def test_evict_layer_behaviors(monkeypatch):
    lm = FakeLayerManagerForCache(FakeModelMetadata({}), [])
    patch_layer_manager_for_cache(monkeypatch, lambda *a, **k: lm)
    wc = WeightCache([0], FakeModelMetadata({}), window_size=1)
    wc.cache[0] = ({"w": mx.array([0.0])}, 0.0)
    wc.reference_counts[0] = 1
    assert wc.evict_layer(0) is False
    wc.reference_counts[0] = 0
    assert wc.evict_layer(12345) is True
    assert wc.evict_layer(0) is True
    assert 0 not in wc.cache


def test_evict_layers_counts(monkeypatch):
    patch_layer_manager_for_cache(monkeypatch, FakeLayerManagerForCache)
    wc = WeightCache([0, 1, 2], FakeModelMetadata({}), window_size=3)
    for lid in (0, 1, 2):
        wc.cache[lid] = ({"w": mx.array([0.0])}, float(lid))
        wc.reference_counts[lid] = 0
    assert wc.evict_layers([0, 2]) == 2


def test_get_weight_evicts_when_full(monkeypatch):
    class _LM(FakeLayerManagerForCache):
        def __init__(self, meta, layers, **k):
            super().__init__(meta, layers, **k)
            self._prefetch_mode = "off"

    patch_layer_manager_for_cache(monkeypatch, _LM)
    meta = FakeModelMetadata({1: {"w": FakeWeightSize(4)}, 2: {"w": FakeWeightSize(4)}})
    wc = WeightCache([1, 2], meta, window_size=1, resident_windows=1)
    d1 = wc.get_weight(1)
    assert d1 is not None
    wc.decrease_reference(
        1
    )  # FIXME: allow eviction by dropping ref count of the first layer
    d2 = wc.get_weight(2)
    assert d2 is not None
    assert 1 not in wc.cache and 2 in wc.cache


def test_prefetch_to_ram_modes_and_reuse(monkeypatch):
    class _PrefLM(FakeLayerManagerForCache):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._prefetch_mode = "sequential"

        def async_prefetch(self, layer_id):
            f = Future()  # leave pending to test reuse
            return f

    patch_layer_manager_for_cache(monkeypatch, _PrefLM)
    wc = WeightCache([0], FakeModelMetadata({}), window_size=1)
    f1 = wc.prefetch_to_ram(0)
    assert isinstance(f1, Future)
    f2 = wc.prefetch_to_ram(0)
    assert f2 is f1


def test_prefetch_to_ram_off_returns_none(monkeypatch):
    class _PrefOff(FakeLayerManagerForCache):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._prefetch_mode = "off"

    patch_layer_manager_for_cache(monkeypatch, _PrefOff)
    wc = WeightCache([0], FakeModelMetadata({}), window_size=1)
    assert wc.prefetch_to_ram(0) is None


def test_cancel_all_prefetch(monkeypatch):
    patch_layer_manager_for_cache(monkeypatch, FakeLayerManagerForCache)
    wc = WeightCache([0], FakeModelMetadata({}), window_size=1)
    f1 = Future()
    f2 = Future()
    wc.prefetch_futures = {0: f1, 1: f2}
    wc.cancel_all_prefetch()
    assert wc.prefetch_futures == {}


def test_get_resident_layers_order():
    wc = WeightCache.__new__(WeightCache)
    wc.lock = threading.Lock()
    wc.cache = {
        5: ({"w": mx.array([0.0])}, 5.0),
        2: ({"w": mx.array([0.0])}, 2.0),
        9: ({"w": mx.array([0.0])}, 9.0),
    }
    out = WeightCache.get_resident_layers(wc)
    assert out == [2, 5, 9]


def test_get_resident_layers_empty():
    wc = WeightCache.__new__(WeightCache)
    wc.lock = threading.Lock()
    wc.cache = {}
    assert WeightCache.get_resident_layers(wc) == []
