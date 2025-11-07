"""Tests for utils.py.

Test separately with:

```sh
uv run pytest -v -k utils_test
```
"""

from dnet_p2p import ThunderboltConnection, DnetDeviceProperties
from distilp.components.dense_common import HALDAResult, DeviceProfile

from .utils import (
    postprocess_single_round,
    optimize_device_ordering,
    compute_layer_assignments,
)


def test_single_round_postprocess_simple():
    device_names = ["dev1", "dev2", "dev3"]
    solution: HALDAResult = HALDAResult(
        k=1,
        #     ____
        w=[4, 1, 3],
        n=[4, 1, 2],
        sets={},  # ignored
        obj_value=0,  # ignored
    )

    new_device_names, new_solution = postprocess_single_round(device_names, solution)
    assert new_device_names == ["dev1", "dev3"]
    assert new_solution.k == 1
    assert new_solution.w == [4, 4]
    assert new_solution.n == [4, 3]


def test_single_round_postprocess_complex():
    device_names = ["dev1", "dev2", "dev3", "dev4", "dev5", "dev6"]
    solution: HALDAResult = HALDAResult(
        k=1,
        # _____     ____
        w=[4, 1, 5, 1, 1, 2],
        n=[4, 1, 4, 1, 1, 1],
        sets={},  # ignored
        obj_value=0,  # ignored
    )

    new_device_names, new_solution = postprocess_single_round(device_names, solution)
    assert new_device_names == ["dev1", "dev3", "dev5", "dev6"]
    assert new_solution.k == 1
    assert new_solution.w == [5, 5, 2, 2]
    assert new_solution.n == [5, 4, 2, 1]


def test_optimize_device_ordering():
    from pydantic import BaseModel

    # fake type for the sake of testing
    class _DeviceProfileIsHead(BaseModel):
        is_head: bool

    device_profiles: dict[str, DeviceProfile] = {
        "dev1": _DeviceProfileIsHead(is_head=False),  # type: ignore
        "dev2": _DeviceProfileIsHead(is_head=False),  # type: ignore
        "dev3": _DeviceProfileIsHead(is_head=True),  # type: ignore
        "dev4": _DeviceProfileIsHead(is_head=False),  # type: ignore
        "dev5": _DeviceProfileIsHead(is_head=False),  # type: ignore
        "dev6": _DeviceProfileIsHead(is_head=False),  # type: ignore
        "dev7": _DeviceProfileIsHead(is_head=False),  # type: ignore
    }
    thunderbolts: dict[str, dict[str, ThunderboltConnection]] = {
        "dev3": {"dev1": 1},  # type: ignore
        "dev1": {"dev3": 1},  # type: ignore
        "dev2": {"dev6": 1},  # type: ignore
        "dev6": {"dev2": 1},  # type: ignore
    }

    optimized_order = optimize_device_ordering(device_profiles, thunderbolts)

    # the ordering is not deterministic, but the connectino should be as follows:
    # head must be the first
    assert optimized_order[0] == "dev3"

    # dev1 and dev3 must be next to each other (due to thunderbolt)
    dev1_index = optimized_order.index("dev1")
    dev3_index = optimized_order.index("dev3")
    assert abs(dev1_index - dev3_index) == 1

    # dev2 and dev6 must be next to each other (due to thunderbolt)
    dev2_index = optimized_order.index("dev2")
    dev6_index = optimized_order.index("dev6")
    assert abs(dev2_index - dev6_index) == 1


def test_layer_assignments_triple():
    # the order here is important
    shard_order = ["dev1", "dev2", "dev3"]
    shards = {
        "dev1": DnetDeviceProperties(
            instance="dev1", local_ip="192.168.0.1", server_port=0, shard_port=0
        ),
        "dev2": DnetDeviceProperties(
            instance="dev2", local_ip="192.168.0.2", server_port=0, shard_port=0
        ),
        "dev3": DnetDeviceProperties(
            instance="dev3", local_ip="192.168.0.3", server_port=0, shard_port=0
        ),
    }

    assignments = compute_layer_assignments(
        shard_order, shards, [3, 5, 4], [3, 4, 2], 2
    )

    # [0]: dev1
    assert assignments[0].instance == "dev1"
    assert assignments[0].layers == [[0, 1, 2], [12, 13, 14]]
    assert assignments[0].residency_size == 3
    assert assignments[0].next_instance == "dev2"

    # [1]: dev2
    assert assignments[1].instance == "dev2"
    assert assignments[1].layers == [[3, 4, 5, 6, 7], [15, 16, 17, 18, 19]]
    assert assignments[1].residency_size == 4
    assert assignments[1].next_instance == "dev3"

    # [2]: dev3
    assert assignments[2].instance == "dev3"
    assert assignments[2].layers == [[8, 9, 10, 11], [20, 21, 22, 23]]
    assert assignments[2].residency_size == 2
    assert assignments[2].next_instance == "dev1"  # wraps around


def test_layer_assignments_single():
    shard_order = ["dev1"]
    shards = {
        "dev1": DnetDeviceProperties(
            instance="dev1", local_ip="192.168.0.1", server_port=0, shard_port=0
        ),
    }

    assignments = compute_layer_assignments(shard_order, shards, [2], [2], 3)

    # [0]: dev1
    assert assignments[0].instance == "dev1"
    assert assignments[0].layers == [[0, 1], [2, 3], [4, 5]]
    assert assignments[0].residency_size == 2
    assert assignments[0].next_instance == "dev1"  # wraps around
