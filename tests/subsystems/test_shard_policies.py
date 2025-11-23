"""Tests: Shard policy registry, make_policy, plan_policy, and helpers."""

import pytest

mx = pytest.importorskip("mlx.core")

from dnet.shard.policies.base import POLICY_REGISTRY, make_policy
from dnet.shard.policies import (
    plan_policy,
    FitInMemoryPolicy,
    OffloadPolicy,
    NoopPolicy,
)
from dnet.shard.config import TopologyConfig
from tests.fakes import FakeRuntimeMinimal

pytestmark = [pytest.mark.shard, pytest.mark.policies]


def test_policy_registry_contains_expected_core_policies():
    assert POLICY_REGISTRY.get("fit") is FitInMemoryPolicy
    assert POLICY_REGISTRY.get("offload") is OffloadPolicy
    assert POLICY_REGISTRY.get("sliding_fit") is OffloadPolicy
    assert POLICY_REGISTRY.get("noop") is NoopPolicy


def test_make_policy_known_and_unknown_modes():
    rt = FakeRuntimeMinimal()
    p1 = make_policy("fit", rt, resident_windows=2)
    assert isinstance(p1, FitInMemoryPolicy)

    p2 = make_policy("offload", rt, resident_windows=3)
    assert isinstance(p2, OffloadPolicy)

    p3 = make_policy("sliding_fit", rt, resident_windows=1)
    assert isinstance(p3, OffloadPolicy)

    p4 = make_policy("noop", rt, resident_windows=0)
    assert isinstance(p4, NoopPolicy)

    with pytest.raises(ValueError):
        _ = make_policy("does_not_exist", rt, resident_windows=1)


def test_plan_policy_variants_fit_offload_and_sliding():
    topo = TopologyConfig()  # resident_windows defaults to 1

    # Fit: requested >= local_count -> fit, window_size == local_count
    plan = plan_policy(
        local_count=4, requested_w=4, residency_size=8, topology_config=topo
    )
    assert plan.mode == "fit" and plan.policy_cls is FitInMemoryPolicy
    assert plan.window_size == 4 and plan.is_sliding is False

    # Offload: partial fit (requested < local_count) and residency large enough
    plan2 = plan_policy(
        local_count=6, requested_w=3, residency_size=5, topology_config=topo
    )
    assert plan2.mode == "offload" and plan2.policy_cls is OffloadPolicy
    assert plan2.window_size == 3 and plan2.is_sliding is False
    assert plan2.resident_windows == topo.resident_windows

    # Sliding-fit: residency smaller than requested -> offload + sliding flag
    plan3 = plan_policy(
        local_count=6, requested_w=4, residency_size=1, topology_config=topo
    )
    assert plan3.mode == "offload" and plan3.policy_cls is OffloadPolicy
    assert plan3.is_sliding is True and plan3.window_size == 1


def test_next_local_layers_helper():
    from dnet.shard.policies.base import ComputePolicy

    s = [1, 2, 3, 5]
    assert ComputePolicy._next_local_layers(s, after_layer=2, count=2) == [3, 5]
    assert ComputePolicy._next_local_layers(s, after_layer=5, count=2) == []
    assert ComputePolicy._next_local_layers(s, after_layer=1, count=0) == []
