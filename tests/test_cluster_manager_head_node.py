"""Tests: ClusterManager head-node selection logic (layer 0 holder)."""

import pytest
from dnet_p2p import DnetDeviceProperties

from dnet.api.cluster import ClusterManager
from dnet.core.types.topology import TopologyInfo, LayerAssignment
from tests.fakes import FakeDiscovery, FakeSolver

pytestmark = [pytest.mark.api]


def test_get_head_node_returns_layer0_holder():
    cm = ClusterManager(FakeDiscovery({}), FakeSolver())
    dev1 = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="A",
        server_port=8001,
        shard_port=9001,
        local_ip="10.0.0.1",
    )
    dev2 = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="B",
        server_port=8002,
        shard_port=9002,
        local_ip="10.0.0.2",
    )
    cm.shards = {"A": dev1, "B": dev2}
    topo = TopologyInfo(
        model="m",
        kv_bits="8bit",
        num_layers=3,
        devices=[dev1, dev2],
        assignments=[
            LayerAssignment(
                instance="B",
                layers=[[1, 2]],
                next_instance=None,
                window_size=1,
                residency_size=1,
            ),
            LayerAssignment(
                instance="A",
                layers=[[0]],
                next_instance="B",
                window_size=1,
                residency_size=1,
            ),
        ],
        solution=None,
    )
    cm.current_topology = topo
    head = cm.get_head_node()
    assert head is dev1


def test_get_head_node_none_when_no_topology():
    cm = ClusterManager(FakeDiscovery({}), FakeSolver())
    assert cm.get_head_node() is None
