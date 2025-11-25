"""Tests: FitInMemoryPolicy and OffloadPolicy configure/process behaviors."""

# ruff: noqa: E402
import asyncio
import pytest

mx = pytest.importorskip("mlx.core")

from dnet.shard.policies.fit_in_memory import FitInMemoryPolicy
from dnet.shard.policies.offload import OffloadPolicy
from dnet.shard.models import ShardLoadModelRequest
from dnet.core.types.messages import ActivationMessage
from tests.fakes import (
    FakeRuntimeForPolicy,
    FakeWeightCache,
    FakeModelMetadata,
    FakeSampler,
)

pytestmark = [pytest.mark.shard, pytest.mark.policies]


def _create_request(window_size=1, residency=2):
    return ShardLoadModelRequest(
        model_path="m",
        total_layers=4,
        layers=[1, 2],
        warmup=False,
        next_node=None,
        window_size=window_size,
        residency_size=residency,
        kv_bits="8bit",
        api_callback_address="",
    )


def _alloc_input(rt: FakeRuntimeForPolicy, layer_id=0, shape=(3,), dtype=mx.float32):
    pid = rt.input_pool.allocate_for_layer(layer_id=layer_id, dtype=dtype, shape=shape)
    buf = rt.input_pool.get_buffer(pid)
    import numpy as np

    buf[: int(np.prod(shape))] = np.arange(0, int(np.prod(shape)), dtype=np.float32)
    return pid


def test_fit_config_uses_weightcache_and_sets_window(monkeypatch):
    rt = FakeRuntimeForPolicy(assigned_layers=[1, 2, 3])
    created = {"args": None}

    def _wc(*a, **k):
        created["args"] = (a, k)
        return FakeWeightCache(*a, **k)

    monkeypatch.setattr(
        "dnet.shard.policies.fit_in_memory.WeightCache", _wc, raising=True
    )

    pol = FitInMemoryPolicy(rt, resident_windows=2)
    pol.configure_policy_for_model(_create_request(window_size=2, residency=3))
    assert pol.window_size == 2 and pol.weight_cache is not None
    assert created["args"] is not None


def test_fit_process_emits_nonfinal_activation(monkeypatch):
    rt = FakeRuntimeForPolicy(assigned_layers=[1, 2], num_layers=10)
    monkeypatch.setattr(
        "dnet.shard.policies.fit_in_memory.WeightCache",
        lambda *a, **k: FakeWeightCache(*a, **k),
        raising=True,
    )
    pol = FitInMemoryPolicy(rt, resident_windows=1)
    pol.configure_policy_for_model(_create_request(window_size=1, residency=2))

    pool_id = _alloc_input(rt, layer_id=0, shape=(3,), dtype=mx.float32)
    msg = ActivationMessage(
        nonce="n1",
        pool_id=pool_id,
        batch_size=1,
        shape=(3,),
        dtype="float32",
        layer_id=0,
        timestamp=0,
        node_origin="S",
        callback_url="",
    )
    pol.process(msg)
    assert rt._emitted, "policy should have emitted one activation"
    out = rt._emitted[-1]
    assert out.is_final is False and out.tensor is not None
    assert str(out.dtype) == str(rt._wire_mx_dtype)


def test_fit_process_end_shard_sampling(monkeypatch):
    rt = FakeRuntimeForPolicy(assigned_layers=[1], num_layers=2)
    monkeypatch.setattr(
        "dnet.shard.policies.fit_in_memory.WeightCache",
        lambda *a, **k: FakeWeightCache(*a, **k),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.policies.fit_in_memory.Sampler", FakeSampler, raising=True
    )

    pol = FitInMemoryPolicy(rt, resident_windows=1)
    pol.configure_policy_for_model(_create_request(window_size=1, residency=1))

    pid = _alloc_input(rt, layer_id=0, shape=(2,), dtype=mx.float32)
    msg = ActivationMessage(
        nonce="n2",
        pool_id=pid,
        batch_size=1,
        shape=(2,),
        dtype="float32",
        layer_id=0,
        timestamp=0,
        node_origin="S",
        callback_url="",
        req_logprobs=True,
        req_top_logprobs=1,
    )
    pol.process(msg)
    assert rt._emitted and rt._emitted[-1].is_final is True
    out = rt._emitted[-1]
    assert int(out.token_id) == 7 and isinstance(out.top_logprobs, dict)


def test_offload_config_sets_mode_and_patches_repack(monkeypatch):
    rt = FakeRuntimeForPolicy(assigned_layers=[1, 2, 3])

    monkeypatch.setattr(
        "dnet.shard.policies.offload.ensure_repacked_for_layers",
        lambda model_path, layers: ("/rep", False),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.policies.offload.get_model_metadata",
        lambda path: FakeModelMetadata(num_layers=6),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.policies.offload.WeightCache",
        lambda *a, **k: FakeWeightCache(*a, **k),
        raising=True,
    )

    pol = OffloadPolicy(rt, resident_windows=1)
    pol.configure_policy_for_model(
        _create_request(window_size=2, residency=5)
    )  # offload mode
    assert pol._mode == "offload" and pol.window_size == 2
    assert (
        rt.compute_config.mxload_fastpath is True
        and rt.compute_config.prefetch_mode == "off"
    )
    assert pol.weight_cache is not None

    # sliding-fit
    pol2 = OffloadPolicy(rt, resident_windows=1)
    pol2.configure_policy_for_model(_create_request(window_size=3, residency=1))
    assert pol2._mode == "sliding_fit" and pol2.window_size == 1


def test_offload_process_schedules_prefetch(monkeypatch):
    rt = FakeRuntimeForPolicy(assigned_layers=[1, 2], num_layers=10)
    monkeypatch.setattr(
        "dnet.shard.policies.offload.ensure_repacked_for_layers",
        lambda *a, **k: ("/rep", False),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.policies.offload.get_model_metadata",
        lambda *a, **k: FakeModelMetadata(num_layers=10),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.policies.offload.WeightCache",
        lambda *a, **k: FakeWeightCache(*a, **k),
        raising=True,
    )

    pol = OffloadPolicy(rt, resident_windows=1)
    pol.configure_policy_for_model(_create_request(window_size=1, residency=2))

    # make _prepare_window_blocking fast
    monkeypatch.setattr(
        pol, "_prepare_window_blocking", lambda layers: True, raising=True
    )

    pid = _alloc_input(rt, layer_id=0, shape=(2,), dtype=mx.float32)
    msg = ActivationMessage(
        nonce="n3",
        pool_id=pid,
        batch_size=1,
        shape=(2,),
        dtype="float32",
        layer_id=0,
        timestamp=0,
        node_origin="S",
        callback_url="",
    )

    async def main():
        # Attach the loop to the runtime so the policy can use it
        rt.attach_loop(asyncio.get_running_loop())
        pol.process(msg)
        # Give the loop a tick to register the future
        await asyncio.sleep(0)

    asyncio.run(main())
    assert rt._emitted, "should emit an activation before scheduling prefetch"
    # Prefetch context stored per nonce
    assert "n3" in pol._prepared_by_nonce
    next_layers, fut = pol._prepared_by_nonce["n3"]
    assert isinstance(next_layers, list) and len(next_layers) >= 1
    assert hasattr(fut, "done")


def test_offload_sliding_eviction_invokes_weightcache_and_unload(monkeypatch):
    rt = FakeRuntimeForPolicy(assigned_layers=[1, 2], num_layers=10)
    monkeypatch.setattr(
        "dnet.shard.policies.offload.ensure_repacked_for_layers",
        lambda *a, **k: ("/rep", False),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.policies.offload.get_model_metadata",
        lambda *a, **k: FakeModelMetadata(num_layers=10),
        raising=True,
    )

    # Use a stateful FakeWeightCache so we can assert evictions
    cache = FakeWeightCache(rt.assigned_layers, rt.model_metadata, window_size=1)
    cache.set_resident_layers([1])
    monkeypatch.setattr(
        "dnet.shard.policies.offload.WeightCache", lambda *a, **k: cache, raising=True
    )

    pol = OffloadPolicy(rt, resident_windows=1)
    # Force sliding_fit: residency (1) < requested window (2)
    pol.configure_policy_for_model(_create_request(window_size=2, residency=1))

    pid = _alloc_input(rt, layer_id=0, shape=(2,), dtype=mx.float32)
    msg = ActivationMessage(
        nonce="n4",
        pool_id=pid,
        batch_size=1,
        shape=(2,),
        dtype="float32",
        layer_id=0,
        timestamp=0,
        node_origin="S",
        callback_url="",
    )
    pol.process(msg)
    # Eviction of previous window (layer 1) should be attempted when moving to layer 2
    assert 1 in cache.evicted
    assert 1 in rt.model.unloaded
