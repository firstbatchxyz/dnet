"""Tests: DynamicMemoryPool allocation, release, and stats behavior."""

import time
import pytest

mx = pytest.importorskip("mlx.core")
from dnet.core.memory.memory_pool import DynamicMemoryPool
from dnet.core.types.messages import PoolStatus

pytestmark = [pytest.mark.core]


def test_init_defaults_and_custom_dynamic_pool():
    p = DynamicMemoryPool()
    assert p.total_memory_bytes == 512 * 1024 * 1024
    assert p.min_buffer_size == 1024
    assert p.used_memory == 0
    assert p.next_buffer_id == 0
    assert hasattr(p.lock, "acquire")
    p2 = DynamicMemoryPool(total_memory_mb=3, min_buffer_size=256)
    assert p2.total_memory_bytes == 3 * 1024 * 1024
    assert p2.min_buffer_size == 256


def test_allocate():
    pool = DynamicMemoryPool(total_memory_mb=2, min_buffer_size=64)
    buf_id = pool.allocate(7, mx.float32)
    assert buf_id is not None
    info = pool.buffer_info[buf_id]
    assert info.size == 8
    assert pool.used_memory == info.size
    st = pool.get_stats()
    assert st["total_buffers"] == 1
    assert st["allocated_buffers"] == 1
    assert st["free_buffers"] == 0


def test_release():
    pool = DynamicMemoryPool(total_memory_mb=2)
    b = pool.allocate(1024, mx.float32)
    assert b is not None
    st1 = pool.get_stats()
    assert st1["allocated_buffers"] == 1
    pool.release(b)
    st2 = pool.get_stats()
    assert st2["allocated_buffers"] == 0
    assert st2["free_buffers"] == 1


def test_multi_buffer_release():
    p = DynamicMemoryPool(total_memory_mb=1)
    b1 = p.allocate(400 * 1024, mx.float32)
    b2 = p.allocate(400 * 1024, mx.float32)
    assert b1 is not None and b2 is not None
    s1 = p.buffer_info[b1].size
    s2 = p.buffer_info[b2].size
    used_before = p.used_memory
    p.release(b1)
    time.sleep(0.005)
    p.release(b2)
    b3 = p.allocate(500 * 1024, mx.float32)
    assert b3 is not None
    assert b1 not in p.buffer_info and b2 not in p.buffer_info
    s3 = p.buffer_info[b3].size
    assert p.used_memory == used_before - (s1 + s2) + s3


# buffers are still removed if alloc fails
def test_multi_buffer_eviction_failure():
    p = DynamicMemoryPool(total_memory_mb=1)
    b1 = p.allocate(100 * 1024, mx.float32)
    b2 = p.allocate(100 * 1024, mx.float32)
    assert b1 is not None and b2 is not None
    p.release(b1)
    p.release(b2)
    used_before = p.used_memory
    next_id_before = p.next_buffer_id
    res = p.allocate(900 * 1024, mx.float32)
    assert res is None
    assert p.used_memory == 0
    assert p.next_buffer_id == next_id_before


def test_reuse_free_buffer_same_size():
    pool = DynamicMemoryPool(total_memory_mb=2)
    b1 = pool.allocate(1024, mx.float32)
    assert b1 is not None
    pool.release(b1)
    b2 = pool.allocate(1024, mx.float32)
    assert b2 == b1


def test_evicts_oldest_free_buffer():
    pool = DynamicMemoryPool(total_memory_mb=1)
    b1 = pool.allocate(200 * 1024, mx.float32)
    assert b1 is not None
    pool.release(b1)
    s1 = pool.buffer_info[b1].size
    time.sleep(0.01)
    b2 = pool.allocate(400 * 1024, mx.float32)
    assert b2 is not None
    pool.release(b2)
    s2 = pool.buffer_info[b2].size
    used_before = pool.used_memory
    big = pool.allocate(500 * 1024, mx.float32)
    assert big is not None
    s_big = pool.buffer_info[big].size
    assert b1 not in pool.buffer_info
    assert b2 not in pool.buffer_info
    assert pool.used_memory == s_big


def test_allocation_failure_when_insufficient():
    pool = DynamicMemoryPool(total_memory_mb=1)
    res = pool.allocate(3 * 1024 * 1024, mx.float32)
    assert res is None


def test_get_buffer_view_success():
    pool = DynamicMemoryPool(total_memory_mb=2)
    buf = pool.allocate(16 * mx.float32.size, mx.float32)
    assert buf is not None
    arr = pool.get_buffer(buf)
    assert arr is not None
    view = pool.get_buffer_view(buf, (4, 4))
    assert view is not None
    assert tuple(view.shape) == (4, 4)


def test_get_buffer_view_failure():
    pool = DynamicMemoryPool(total_memory_mb=2)
    buf = pool.allocate(16 * mx.float32.size, mx.float32)
    too_big = pool.get_buffer_view(buf, (5, 5))
    assert too_big is None


def test_allocate_invalid_arguments():
    p = DynamicMemoryPool(total_memory_mb=1)
    with pytest.raises(Exception):
        _ = p.allocate(128, None)  # type: ignore
    assert p.used_memory == 0
    assert p.get_buffer(0) is None
    b = p.allocate(128, mx.float32)  # type: ignore
    with pytest.raises(Exception):
        v = p.get_buffer_view(b, (2, "x"))  # type: ignore


def test_post_release_access_denied_and_last_used_updates():
    p = DynamicMemoryPool(total_memory_mb=2)
    b = p.allocate(64, mx.float32)
    assert b is not None
    t0 = p.buffer_info[b].last_used
    time.sleep(0.005)
    _ = p.get_buffer(b)
    assert p.buffer_info[b].last_used > t0
    p.release(b)
    assert p.get_buffer(b) is None
    assert p.get_buffer_view(b, (1, 1)) is None


def test_no_reuse_different_size():
    p = DynamicMemoryPool(total_memory_mb=2)
    a = p.allocate(256, mx.float32)
    assert a is not None
    p.release(a)
    b = p.allocate(128, mx.float32)
    assert b is not None
    assert b != a


def test_dtype_alignment_mixed():
    p = DynamicMemoryPool(total_memory_mb=2)
    b1 = p.allocate(5, mx.float16)
    b2 = p.allocate(5, mx.float32)
    assert b1 is not None and b2 is not None
    s1 = p.buffer_info[b1].size
    s2 = p.buffer_info[b2].size
    assert s1 % mx.float16.size == 0 and s2 % mx.float32.size == 0
    assert s1 == 6 and s2 == 8


# overwrite mx.zeros to raies
def test_memoryerror_does_not_mutate_state(monkeypatch):
    from dnet.core.memory import memory_pool as mpmod

    p = DynamicMemoryPool(total_memory_mb=1)
    before_used = p.used_memory
    before_next = p.next_buffer_id
    real_zeros = mpmod.mx.zeros

    def boom(*a, **k):
        raise MemoryError

    monkeypatch.setattr(mpmod.mx, "zeros", boom)
    try:
        res = p.allocate(128, mx.float32)
        assert res is None
        assert p.used_memory == before_used
        assert p.next_buffer_id == before_next
    finally:
        monkeypatch.setattr(mpmod.mx, "zeros", real_zeros)


def test_size_to_buffers_integrity_after_eviction():
    p = DynamicMemoryPool(total_memory_mb=1)
    b1 = p.allocate(200 * 1024, mx.float32)
    b2 = p.allocate(100 * 1024, mx.float32)
    assert b1 is not None and b2 is not None
    s1 = p.buffer_info[b1].size
    s2 = p.buffer_info[b2].size
    p.release(b1)
    time.sleep(0.005)
    p.release(b2)
    m = p.size_to_buffers
    assert b1 in m.get(s1, []) and b2 in m.get(s2, [])
    _ = p.allocate(800 * 1024, mx.float32)
    assert b1 not in p.buffer_info and b2 not in p.buffer_info
    assert b1 not in m.get(s1, []) and b2 not in m.get(s2, [])


def test_internal_find_free_buffer():
    p = DynamicMemoryPool(total_memory_mb=2)
    b = p.allocate(64, mx.float32)
    assert b is not None
    size = p.buffer_info[b].size
    assert p._find_free_buffer(size) is None
    p.release(b)
    found = p._find_free_buffer(size)
    assert found == b
    assert p._find_free_buffer(size - mx.float32.size) is None


def test_internal_evict_unused_buffers_only_frees_free():
    p = DynamicMemoryPool(total_memory_mb=1)
    a = p.allocate(200 * 1024, mx.float32)
    b = p.allocate(200 * 1024, mx.float32)
    assert a is not None and b is not None
    p.release(a)
    s_alloc_before = set(p.buffer_info.keys())
    ok = p._evict_unused_buffers(100 * 1024)
    assert ok is True
    assert a not in p.buffer_info
    assert b in p.buffer_info
    assert set(p.buffer_info.keys()).issubset(s_alloc_before)
    assert p.buffer_info[b].status == PoolStatus.ALLOCATED


def test_internal_evict_returns_false_when_insufficient_and_keeps_allocated():
    p = DynamicMemoryPool(total_memory_mb=1)
    a = p.allocate(300 * 1024, mx.float32)
    c = p.allocate(300 * 1024, mx.float32)
    assert a is not None and c is not None
    p.release(a)
    ok = p._evict_unused_buffers(900 * 1024)
    assert ok is False
    assert a not in p.buffer_info
    assert c in p.buffer_info
