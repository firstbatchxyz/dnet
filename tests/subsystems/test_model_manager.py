"""Tests: ModelManager orchestration of shard load/unload over HTTP."""

import asyncio
import pytest

from dnet_p2p import DnetDeviceProperties

from dnet.api.model_manager import ModelManager
from dnet.core.types.topology import TopologyInfo, LayerAssignment
from tests.fakes import FakeClient, FakeResponse

# Mark this module as API
pytestmark = [pytest.mark.api]


def _mk_topology():
    dev1 = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="S1",
        server_port=8011,
        shard_port=9011,
        local_ip="10.0.0.1",
    )
    dev2 = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="S2",
        server_port=8012,
        shard_port=9012,
        local_ip="10.0.0.2",
    )
    topo = TopologyInfo(
        model="m",
        kv_bits="8bit",
        num_layers=4,
        devices=[dev1, dev2],
        assignments=[
            LayerAssignment(
                instance="S1",
                layers=[[0, 1]],
                next_instance="S2",
                window_size=2,
                residency_size=2,
            ),
            LayerAssignment(
                instance="S2",
                layers=[[2, 3]],
                next_instance=None,
                window_size=2,
                residency_size=2,
            ),
        ],
        solution=None,
    )
    return topo, dev1, dev2


def test_load_model_success_posts_and_tokenizer_loaded(monkeypatch):
    topo, dev1, dev2 = _mk_topology()
    mm = ModelManager()

    rec = {}

    def _mk_post(url):
        def f(payload):
            rec[url] = payload
            return FakeResponse(
                200,
                {
                    "success": True,
                    "message": "ok",
                    "layers_loaded": payload["layers"],
                    "load_time_ms": 1.0,
                },
            )

        return f

    post_map = {
        f"http://{dev1.local_ip}:{dev1.server_port}/load_model": _mk_post(
            f"http://{dev1.local_ip}:{dev1.server_port}/load_model"
        ),
        f"http://{dev2.local_ip}:{dev2.server_port}/load_model": _mk_post(
            f"http://{dev2.local_ip}:{dev2.server_port}/load_model"
        ),
    }
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda: FakeClient({}, post_map), raising=True
    )
    monkeypatch.setattr(
        "dnet.api.model_manager.resolve_tokenizer_dir",
        lambda m: "/tmp/dir",
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.api.model_manager.load_tokenizer", lambda d, cfg: object(), raising=True
    )

    api_props = DnetDeviceProperties(
        is_manager=True,
        is_busy=False,
        instance="API",
        server_port=0,
        shard_port=0,
        local_ip="1.1.1.1",
    )

    async def main():
        res = await mm.load_model(topo, api_props, grpc_port=5050)
        assert res.success is True
        assert mm.current_model_id == "m" and mm.tokenizer is not None
        # Verify posted payloads
        p1 = rec[f"http://{dev1.local_ip}:{dev1.server_port}/load_model"]
        assert p1["model_path"] == "m"
        assert p1["total_layers"] == 4
        assert p1["layers"] == [0, 1]
        assert p1["warmup"] is True
        assert p1["window_size"] == 2 and p1["residency_size"] == 2
        assert p1["kv_bits"] == "8bit"
        assert p1["api_callback_address"] == "1.1.1.1:5050"
        assert isinstance(p1["next_node"], dict) and p1["next_node"]["instance"] == "S2"
        p2 = rec[f"http://{dev2.local_ip}:{dev2.server_port}/load_model"]
        assert p2["next_node"] is None

    asyncio.run(main())


def test_load_model_partial_failure_no_tokenizer(monkeypatch):
    topo, dev1, dev2 = _mk_topology()
    mm = ModelManager()

    def ok(payload):
        return FakeResponse(
            200,
            {
                "success": True,
                "message": "ok",
                "layers_loaded": payload["layers"],
                "load_time_ms": 1.0,
            },
        )

    def fail(payload):
        return FakeResponse(
            200,
            {
                "success": False,
                "message": "boom",
                "layers_loaded": [],
                "load_time_ms": 1.0,
            },
        )

    post_map = {
        f"http://{dev1.local_ip}:{dev1.server_port}/load_model": ok,
        f"http://{dev2.local_ip}:{dev2.server_port}/load_model": fail,
    }
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda: FakeClient({}, post_map), raising=True
    )
    # ensure tokenizer path is not hit
    monkeypatch.setattr(
        "dnet.api.model_manager.resolve_tokenizer_dir",
        lambda m: (_ for _ in ()).throw(AssertionError("should not load tokenizer")),
        raising=True,
    )

    api_props = DnetDeviceProperties(
        is_manager=True,
        is_busy=False,
        instance="API",
        server_port=0,
        shard_port=0,
        local_ip="1.1.1.1",
    )

    async def main():
        res = await mm.load_model(topo, api_props, grpc_port=5050)
        assert res.success is False
        assert mm.current_model_id is None and mm.tokenizer is None

    asyncio.run(main())


def test_unload_model_success_and_failure(monkeypatch):
    mm = ModelManager()
    mm.current_model_id = "m"
    shards = {
        "S1": DnetDeviceProperties(
            is_manager=False,
            is_busy=False,
            instance="S1",
            server_port=8011,
            shard_port=9011,
            local_ip="10.0.0.1",
        ),
        "M": DnetDeviceProperties(
            is_manager=True,
            is_busy=False,
            instance="M",
            server_port=8000,
            shard_port=9000,
            local_ip="127.0.0.1",
        ),
    }
    post_map = {
        "http://10.0.0.1:8011/unload_model": (
            lambda payload: FakeResponse(200, {"ok": True})
        ),
    }
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda: FakeClient({}, post_map), raising=True
    )

    async def main():
        res = await mm.unload_model(shards)
        assert res.success is True
        assert mm.current_model_id is None and mm.tokenizer is None

    asyncio.run(main())


def test_load_model_non200_response_causes_failure(monkeypatch):
    topo, dev1, _ = _mk_topology()
    mm = ModelManager()

    # One shard returns 500 with invalid payload to trigger validation error
    post_map = {
        f"http://{dev1.local_ip}:{dev1.server_port}/load_model": (
            lambda payload: FakeResponse(500, {})
        ),
    }
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda: FakeClient({}, post_map), raising=True
    )
    # Ensure tokenizer load is not called
    called = {"tok": False}

    def _tok_dir(m):
        called["tok"] = True
        return "/tmp"

    monkeypatch.setattr(
        "dnet.api.model_manager.resolve_tokenizer_dir", _tok_dir, raising=True
    )

    api_props = DnetDeviceProperties(
        is_manager=True,
        is_busy=False,
        instance="API",
        server_port=0,
        shard_port=0,
        local_ip="1.1.1.1",
    )

    async def main():
        res = await mm.load_model(topo, api_props, grpc_port=5050)
        assert res.success is False
        assert mm.current_model_id is None

    asyncio.run(main())
