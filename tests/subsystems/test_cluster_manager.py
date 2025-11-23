"""Tests: ClusterManager discovery, solver/plumbing, and HTTP client flows."""

import asyncio
import pytest
import types
import logging

from dnet.api.cluster import ClusterManager
from tests.fakes import (
    FakeDiscovery,
    FakeSolver,
    FakeClient,
    FakeResponse,
    FakeLatencyRequest,
    FakeProfileRequest,
    FakeLatencyResponse,
    FakeProfileResponse,
)

# Silence logs
logger = logging.getLogger("dnet")
logger.setLevel(logging.CRITICAL)

pytestmark = [pytest.mark.api]


def _monkey_models(monkeypatch, median_val):
    monkeypatch.setattr(
        "dnet.api.cluster.MeasureLatencyRequest", FakeLatencyRequest, raising=True
    )
    monkeypatch.setattr(
        "dnet.api.cluster.ShardProfileRequest", FakeProfileRequest, raising=True
    )
    monkeypatch.setattr(
        "dnet.api.cluster.MeasureLatencyResponse",
        types.SimpleNamespace(
            model_validate=lambda p: FakeLatencyResponse(
                p.get("latency") if isinstance(p, dict) else {"k": 1}
            )
        ),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.api.cluster.ShardProfileResponse",
        types.SimpleNamespace(model_validate=lambda p: FakeProfileResponse()),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.api.cluster.calculate_median_latency_seconds",
        lambda lat: median_val,
        raising=True,
    )


def _fake_urls(shards):
    urls = {}
    for n, s in shards.items():
        urls[f"http://{s.local_ip}:{s.server_port}/health"] = n
        urls[f"http://{s.local_ip}:{s.server_port}/measure_latency"] = n
        urls[f"http://{s.local_ip}:{s.server_port}/profile"] = n
    return urls


def test_profile_all_healthy_no_tb(monkeypatch):
    # Make Fake discovery
    shards = {
        "A": {
            "instance": "A",
            "local_ip": "10.0.0.1",
            "server_port": 8001,
            "is_manager": False,
        },
        "B": {
            "instance": "B",
            "local_ip": "10.0.0.2",
            "server_port": 8002,
            "is_manager": False,
        },
    }
    disc = FakeDiscovery(shards)
    cm = ClusterManager(disc, FakeSolver())

    # Fake Thunderbolt
    monkeypatch.setattr(
        "dnet.api.cluster.discover_all_thunderbolt_connections",
        lambda x: {},
        raising=True,
    )
    _monkey_models(monkeypatch, median_val=0.5)

    # Fake httpx.AsyncClient
    get_map = {
        f"http://{shards[k]['local_ip']}:{shards[k]['server_port']}/health": (
            lambda: FakeResponse(200)
        )
        for k in shards
    }
    post_map = {
        f"http://{shards[k]['local_ip']}:{shards[k]['server_port']}/measure_latency": (
            lambda payload: FakeResponse(200, {"latency": {"L": [1, 2, 3]}})
        )
        for k in shards
    }
    post_map.update(
        {
            f"http://{shards[k]['local_ip']}:{shards[k]['server_port']}/profile": (
                lambda payload: FakeResponse(200, {"ok": True})
            )
            for k in shards
        }
    )
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda: FakeClient(get_map, post_map), raising=True
    )

    async def run():
        await cm.scan_devices()
        profs = await cm.profile_cluster(
            "m", embedding_size=4, max_batch_exp=1, batch_sizes=[1]
        )
        assert set(profs.keys()) == set(shards.keys())
        for v in profs.values():
            assert hasattr(v, "t_comm") and v.t_comm == 0.5

    asyncio.run(run())


def test_profile_some_tb_and_some_health_fail(monkeypatch):
    shards = {
        "A": {
            "instance": "A",
            "local_ip": "10.0.0.1",
            "server_port": 8001,
            "is_manager": False,
        },
        "B": {
            "instance": "B",
            "local_ip": "10.0.0.2",
            "server_port": 8002,
            "is_manager": False,
        },
        "C": {
            "instance": "C",
            "local_ip": "10.0.0.3",
            "server_port": 8003,
            "is_manager": False,
        },
    }
    disc = FakeDiscovery(shards)
    cm = ClusterManager(disc, FakeSolver())

    monkeypatch.setattr(
        "dnet.api.cluster.discover_all_thunderbolt_connections",
        lambda x: {"A": {"B": {"link": True}}},
        raising=True,
    )
    _monkey_models(monkeypatch, median_val=0.1)

    get_map = {
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/health": (
            lambda: FakeResponse(200)
        ),
        f"http://{shards['B']['local_ip']}:{shards['B']['server_port']}/health": (
            lambda: FakeResponse(500)
        ),
        f"http://{shards['C']['local_ip']}:{shards['C']['server_port']}/health": (
            lambda: FakeResponse(200)
        ),
    }
    post_map = {
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/measure_latency": (
            lambda payload: FakeResponse(200, {"latency": {"L": [1]}})
        ),
        f"http://{shards['C']['local_ip']}:{shards['C']['server_port']}/measure_latency": (
            lambda payload: FakeResponse(200, {"latency": {"L": [2]}})
        ),
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/profile": (
            lambda payload: FakeResponse(200, {"ok": True})
        ),
        f"http://{shards['C']['local_ip']}:{shards['C']['server_port']}/profile": (
            lambda payload: FakeResponse(200, {"ok": True})
        ),
    }
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda: FakeClient(get_map, post_map), raising=True
    )

    async def run():
        await cm.scan_devices()
        profs = await cm.profile_cluster(
            "m", embedding_size=4, max_batch_exp=1, batch_sizes=[1]
        )
        assert set(profs.keys()) == {"A", "C"}
        for v in profs.values():
            assert v.t_comm == 0.1

    asyncio.run(run())


def test_profile_no_healthy_returns_empty(monkeypatch):
    shards = {
        "A": {
            "instance": "A",
            "local_ip": "10.0.0.1",
            "server_port": 8001,
            "is_manager": False,
        }
    }
    disc = FakeDiscovery(shards)
    cm = ClusterManager(disc, FakeSolver())
    monkeypatch.setattr(
        "dnet.api.cluster.discover_all_thunderbolt_connections",
        lambda x: {},
        raising=True,
    )
    _monkey_models(monkeypatch, median_val=None)
    get_map = {
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/health": (
            lambda: FakeResponse(500)
        )
    }
    post_map = {}
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda: FakeClient(get_map, post_map), raising=True
    )

    async def run():
        await cm.scan_devices()
        profs = await cm.profile_cluster(
            "m", embedding_size=4, max_batch_exp=1, batch_sizes=[1]
        )
        assert profs == {}

    asyncio.run(run())


def test_solve_topology_sets_current():
    shards = {}
    cm = ClusterManager(FakeDiscovery(shards), FakeSolver())

    async def run():
        out = await cm.solve_topology({}, {}, "m", 1, "fp16")
        assert out["ok"] is True
        assert cm.current_topology == out

    asyncio.run(run())


def test_profile_managers_only_returns_empty(monkeypatch):
    shards = {
        "M": {
            "instance": "M",
            "local_ip": "127.0.0.1",
            "server_port": 8000,
            "is_manager": True,
        },
    }
    disc = FakeDiscovery(shards)
    cm = ClusterManager(disc, FakeSolver())
    monkeypatch.setattr(
        "dnet.api.cluster.discover_all_thunderbolt_connections",
        lambda x: {},
        raising=True,
    )
    _monkey_models(monkeypatch, median_val=0.5)
    # No health calls are made since all are managers; but provide empty maps anyway
    monkeypatch.setattr("httpx.AsyncClient", lambda: FakeClient({}, {}), raising=True)

    async def run():
        await cm.scan_devices()
        profs = await cm.profile_cluster(
            "m", embedding_size=4, max_batch_exp=1, batch_sizes=[1]
        )
        assert profs == {}

    asyncio.run(run())


def test_profile_latency_non_200_drops_all(monkeypatch):
    shards = {
        "A": {
            "instance": "A",
            "local_ip": "10.0.0.1",
            "server_port": 8001,
            "is_manager": False,
        },
        "B": {
            "instance": "B",
            "local_ip": "10.0.0.2",
            "server_port": 8002,
            "is_manager": False,
        },
    }
    disc = FakeDiscovery(shards)
    cm = ClusterManager(disc, FakeSolver())
    monkeypatch.setattr(
        "dnet.api.cluster.discover_all_thunderbolt_connections",
        lambda x: {},
        raising=True,
    )
    _monkey_models(monkeypatch, median_val=0.1)
    get_map = {
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/health": (
            lambda: FakeResponse(200)
        ),
        f"http://{shards['B']['local_ip']}:{shards['B']['server_port']}/health": (
            lambda: FakeResponse(200)
        ),
    }
    post_map = {
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/measure_latency": (
            lambda payload: FakeResponse(500, {})
        ),
        f"http://{shards['B']['local_ip']}:{shards['B']['server_port']}/measure_latency": (
            lambda payload: FakeResponse(500, {})
        ),
    }
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda: FakeClient(get_map, post_map), raising=True
    )

    async def run():
        await cm.scan_devices()
        profs = await cm.profile_cluster(
            "m", embedding_size=4, max_batch_exp=1, batch_sizes=[1]
        )
        assert profs == {}

    asyncio.run(run())


def test_profile_profile_partial_fail(monkeypatch):
    shards = {
        "A": {
            "instance": "A",
            "local_ip": "10.0.0.1",
            "server_port": 8001,
            "is_manager": False,
        },
        "B": {
            "instance": "B",
            "local_ip": "10.0.0.2",
            "server_port": 8002,
            "is_manager": False,
        },
    }
    disc = FakeDiscovery(shards)
    cm = ClusterManager(disc, FakeSolver())
    monkeypatch.setattr(
        "dnet.api.cluster.discover_all_thunderbolt_connections",
        lambda x: {},
        raising=True,
    )
    _monkey_models(monkeypatch, median_val=0.2)
    get_map = {
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/health": (
            lambda: FakeResponse(200)
        ),
        f"http://{shards['B']['local_ip']}:{shards['B']['server_port']}/health": (
            lambda: FakeResponse(200)
        ),
    }
    post_map = {
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/measure_latency": (
            lambda payload: FakeResponse(200, {"latency": {"L": [1]}})
        ),
        f"http://{shards['B']['local_ip']}:{shards['B']['server_port']}/measure_latency": (
            lambda payload: FakeResponse(200, {"latency": {"L": [2]}})
        ),
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/profile": (
            lambda payload: FakeResponse(200, {"ok": True})
        ),
        f"http://{shards['B']['local_ip']}:{shards['B']['server_port']}/profile": (
            lambda payload: FakeResponse(500, {})
        ),
    }
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda: FakeClient(get_map, post_map), raising=True
    )

    async def run():
        await cm.scan_devices()
        profs = await cm.profile_cluster(
            "m", embedding_size=4, max_batch_exp=1, batch_sizes=[1]
        )
        assert set(profs.keys()) == {"A"}
        assert profs["A"].t_comm == 0.2

    asyncio.run(run())


def test_profile_median_none_keeps_default_t_comm(monkeypatch):
    shards = {
        "A": {
            "instance": "A",
            "local_ip": "10.0.0.1",
            "server_port": 8001,
            "is_manager": False,
        },
    }
    disc = FakeDiscovery(shards)
    cm = ClusterManager(disc, FakeSolver())
    monkeypatch.setattr(
        "dnet.api.cluster.discover_all_thunderbolt_connections",
        lambda x: {},
        raising=True,
    )
    _monkey_models(monkeypatch, median_val=None)
    get_map = {
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/health": (
            lambda: FakeResponse(200)
        )
    }
    post_map = {
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/measure_latency": (
            lambda payload: FakeResponse(200, {"latency": {"L": [1]}})
        ),
        f"http://{shards['A']['local_ip']}:{shards['A']['server_port']}/profile": (
            lambda payload: FakeResponse(200, {"ok": True})
        ),
    }
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda: FakeClient(get_map, post_map), raising=True
    )

    async def run():
        await cm.scan_devices()
        profs = await cm.profile_cluster(
            "m", embedding_size=4, max_batch_exp=1, batch_sizes=[1]
        )
        assert "A" in profs and getattr(profs["A"], "t_comm", 0.0) == 0.0

    asyncio.run(run())


def test_solve_topology_raises_on_failure():
    from tests.fakes import FakeBadSolver

    cm = ClusterManager(FakeDiscovery({}), FakeBadSolver())

    async def run():
        with pytest.raises(RuntimeError):
            await cm.solve_topology({}, {}, "m", 1, "fp16")

    asyncio.run(run())
