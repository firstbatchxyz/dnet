"""Tests: LayerAwareMemoryPool per-layer alloc/stats, views, and input validation."""

import pytest

mx = pytest.importorskip("mlx.core")
from dnet.core.memory.memory_pool import LayerAwareMemoryPool

pytestmark = [pytest.mark.core]


def test_layer_aware_init_defaults():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    assert lap.pool is not None
    assert lap.layer_stats == {}


def test_layer_aware_allocation_and_typical_size():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    b1 = lap.allocate_for_layer(0, (2, 4), mx.float32)
    assert b1 is not None
    b2 = lap.allocate_for_layer(0, (2, 6), mx.float32)
    assert b2 is not None
    b3 = lap.allocate_for_layer(0, (2, 2), mx.float32)
    assert b3 is not None
    typical = lap.get_typical_size(0)
    assert typical == 32
    stats = lap.get_stats()
    assert "pool" in stats and "layer_stats" in stats
    assert 0 in stats["layer_stats"]
    layer0 = stats["layer_stats"][0]
    assert layer0["allocations"] >= 3


def test_layer_aware_get_layer_buffer_and_release():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    shape = (3, 3)
    b = lap.allocate_for_layer(1, shape, mx.float32)
    assert b is not None
    view = lap.get_layer_buffer(b, shape)
    assert view is not None
    assert tuple(view.shape) == shape
    lap.release(b)
    ps = lap.pool.get_stats()
    assert ps["free_buffers"] >= 1


def test_layer_aware_invalid_inputs():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    with pytest.raises(Exception):
        _ = lap.allocate_for_layer(0, (2, "x"), mx.float32)  # type: ignore
    with pytest.raises(Exception):
        _ = lap.allocate_for_layer(0, (2, 2), None)  # type: ignore


def test_layer_aware_typical_size_none_for_unknown():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    assert lap.get_typical_size(999) is None


def test_layer_aware_wrong_shape_view_returns_none():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    b = lap.allocate_for_layer(2, (2, 2), mx.float32)
    assert b is not None
    assert lap.get_layer_buffer(b, (3, 3)) is None
    lap.release(b)


def test_layer_aware_trims_stats_lists():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    for i in range(120):
        b = lap.allocate_for_layer(3, (1, 1), mx.float32)
        assert b is not None
        lap.release(b)
    s = lap.layer_stats[3]
    assert 50 <= len(s["sizes"]) <= 100
    assert 50 <= len(s["shapes"]) <= 100


def test_layer_aware_recent_shapes_and_avg_size():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    for i in range(15):
        b = lap.allocate_for_layer(4, (1, (i % 5) + 1), mx.float32)
        assert b is not None
        lap.release(b)
    st = lap.get_stats()
    rec = st["layer_stats"][4]["recent_shapes"]
    assert len(rec) <= 10
    assert st["layer_stats"][4]["avg_size_mb"] > 0.0


def test_layer_aware_allocation_failure_and_stats_increment():
    lap = LayerAwareMemoryPool(total_memory_mb=1)
    big = (1024, 1024)
    bid = lap.allocate_for_layer(7, big, mx.float32)
    assert bid is None
    assert 7 in lap.layer_stats
    assert lap.layer_stats[7]["allocations"] == 1


def test_layer_aware_cross_layer_isolation():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    a = lap.allocate_for_layer(10, (1, 8), mx.float32)
    assert a is not None
    b = lap.allocate_for_layer(20, (1, 2), mx.float32)
    assert b is not None
    ta = lap.get_typical_size(10)
    tb = lap.get_typical_size(20)
    assert ta != tb


def test_layer_aware_even_count_typical_upper_median():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    for n in (1, 2, 3, 4):
        bid = lap.allocate_for_layer(11, (1, n), mx.float32)
        assert bid is not None
        lap.release(bid)
    assert lap.get_typical_size(11) == 12


def test_layer_aware_get_buffer_passthrough():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    b = lap.allocate_for_layer(12, (2, 2), mx.float32)
    assert b is not None
    raw = lap.get_buffer(b)
    assert raw is not None
    lap.release(b)


def test_layer_aware_release_invalid_id_no_crash():
    lap = LayerAwareMemoryPool(total_memory_mb=2)
    lap.release(123456)
