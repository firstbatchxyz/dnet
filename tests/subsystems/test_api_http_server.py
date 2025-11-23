"""Tests: API HTTP server endpoints (health, topology, load/unload, chat, profile)."""

import json
import asyncio
import pytest
from fastapi.testclient import TestClient

from dnet_p2p import DnetDeviceProperties

from dnet.core.types.topology import LayerAssignment, TopologyInfo
from dnet.api.http_api import HTTPServer

from tests.fakes import (
    FakeClusterManager,
    FakeInferenceManager,
    FakeModelManager,
    FakeProps,
)

pytestmark = [pytest.mark.api, pytest.mark.http]


def _create_server(cm, im, mm, port=18080):
    srv = HTTPServer(port, cm, im, mm, node_id="node-A")
    asyncio.run(srv._setup_routes())
    return srv


def test_health_and_devices():
    shards = {
        "A": FakeProps("A", "127.0.0.1", 8011, is_manager=False),
        "B": FakeProps("B", "127.0.0.2", 8012, is_manager=True),
    }
    cm = FakeClusterManager(shards)
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok"
        assert h["http_port"] == srv.http_port
        assert h["model_loaded"] is False
        d = client.get("/v1/devices").json()
        assert "devices" in d and "A" in d["devices"]


def test_chat_completions_non_streaming_success():
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id="m")
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
        }
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["message"]["content"] == "ok"


def test_chat_completions_streaming_sse():
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id="m")
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
            "stream": True,
        }
        with client.stream("POST", "/v1/chat/completions", json=payload) as r:
            assert r.status_code == 200
            lines = [
                ln for ln in r.iter_lines() if ln
            ]  # expect at least two data frames and a DONE line
            assert any(ln.startswith("data: {") for ln in lines)
            assert any(ln.strip() == "data: [DONE]" for ln in lines)
            first_data = next(ln for ln in lines if ln.startswith("data: {"))
            obj = json.loads(first_data[len("data: ") :])
            choice = obj["choices"][0]
            assert "delta" in choice and "message" not in choice


def test_chat_completions_503_when_no_model():
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
        }
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 503


def test_prepare_topology_manual_success():
    cm = FakeClusterManager({})
    im = FakeInferenceManager(grpc_port=50505)
    mm = FakeModelManager(current_model_id=None, load_success=True)
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        devices = [
            {
                "instance": "S1",
                "local_ip": "10.0.0.1",
                "server_port": 8001,
                "shard_port": 9001,
            },
            {
                "instance": "S2",
                "local_ip": "10.0.0.2",
                "server_port": 8002,
                "shard_port": 9002,
            },
        ]
        assignments = [
            {
                "instance": "S1",
                "layers": [[0, 1]],
                "next_instance": "S2",
                "window_size": 1,
                "residency_size": 1,
            },
            {
                "instance": "S2",
                "layers": [[2, 3]],
                "next_instance": None,
                "window_size": 1,
                "residency_size": 1,
            },
        ]
        req = {
            "model": "m",
            "kv_bits": "8bit",
            "devices": devices,
            "assignments": assignments,
            "num_layers": 4,
        }
        r = client.post("/v1/prepare_topology_manual", json=req)
        assert r.status_code == 200
        topo = r.json()
        assert topo["num_layers"] == 4 and topo["model"] == "m"


def test_get_topology_after_prepare_returns_last_topology():
    cm = FakeClusterManager({})
    im = FakeInferenceManager(grpc_port=50505)
    mm = FakeModelManager(current_model_id=None, load_success=True)
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        devices = [
            {
                "instance": "S1",
                "local_ip": "10.0.0.1",
                "server_port": 8001,
                "shard_port": 9001,
            },
            {
                "instance": "S2",
                "local_ip": "10.0.0.2",
                "server_port": 8002,
                "shard_port": 9002,
            },
        ]
        assignments = [
            {
                "instance": "S1",
                "layers": [[0, 1]],
                "next_instance": "S2",
                "window_size": 1,
                "residency_size": 1,
            },
            {
                "instance": "S2",
                "layers": [[2, 3]],
                "next_instance": None,
                "window_size": 1,
                "residency_size": 1,
            },
        ]
        req = {
            "model": "m",
            "kv_bits": "8bit",
            "devices": devices,
            "assignments": assignments,
            "num_layers": 4,
        }
        assert client.post("/v1/prepare_topology_manual", json=req).status_code == 200
        r2 = client.get("/v1/topology")
        assert r2.status_code == 200
        topo = r2.json()
        assert topo["model"] == "m" and topo["num_layers"] == 4


def test_load_model_with_manual_topology_connects_first_shard():
    cm = FakeClusterManager({})
    im = FakeInferenceManager(grpc_port=50505)
    mm = FakeModelManager(current_model_id=None, load_success=True)
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        devices = [
            {
                "instance": "S1",
                "local_ip": "10.0.0.1",
                "server_port": 8001,
                "shard_port": 9001,
            },
            {
                "instance": "S2",
                "local_ip": "10.0.0.2",
                "server_port": 8002,
                "shard_port": 9002,
            },
        ]
        assignments = [
            {
                "instance": "S1",
                "layers": [[0, 1]],
                "next_instance": "S2",
                "window_size": 1,
                "residency_size": 1,
            },
            {
                "instance": "S2",
                "layers": [[2, 3]],
                "next_instance": None,
                "window_size": 1,
                "residency_size": 1,
            },
        ]
        req = {
            "model": "m",
            "kv_bits": "8bit",
            "devices": devices,
            "assignments": assignments,
            "num_layers": 4,
        }
        assert client.post("/v1/prepare_topology_manual", json=req).status_code == 200
        r3 = client.post("/v1/load_model", json={"model": None, "kv_bits": "8bit"})
        assert r3.status_code == 200 and r3.json()["success"] is True
        assert (
            im.connected is not None
            and im.connected[0] == "10.0.0.1"
            and im.connected[1] == 9001
        )


def test_unload_model_sets_topology_none_on_success():
    shards = {
        "S1": FakeProps("S1", "127.0.0.1", 8001, is_manager=False),
        "M": FakeProps("M", "127.0.0.2", 8000, is_manager=True),
    }
    cm = FakeClusterManager(shards)
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id="m", unload_success=True)
    srv = _create_server(cm, im, mm)
    cm.current_topology = TopologyInfo(
        model="m",
        kv_bits="8bit",
        num_layers=1,
        devices=[],
        assignments=[],
        solution=None,
    )
    with TestClient(srv.app) as client:
        r = client.post("/v1/unload_model")
        assert r.status_code == 200
        assert cm.current_topology is None


def test_unload_model_failure_keeps_topology():
    shards = {"S1": FakeProps("S1", "127.0.0.1", 8001, is_manager=False)}
    cm = FakeClusterManager(shards)
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id="m", unload_success=False)
    srv = _create_server(cm, im, mm)
    cm.current_topology = TopologyInfo(
        model="m",
        kv_bits="8bit",
        num_layers=1,
        devices=[],
        assignments=[],
        solution=None,
    )
    with TestClient(srv.app) as client:
        r = client.post("/v1/unload_model")
        assert r.status_code == 200
        assert cm.current_topology is not None


def test_get_topology_400_when_none():
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        r = client.get("/v1/topology")
        assert r.status_code == 400


def test_prepare_topology_profiles_empty_500(monkeypatch):
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = _create_server(cm, im, mm)

    # mock model config + empty profiling
    monkeypatch.setattr(
        "dnet.api.http_api.get_model_config_json",
        lambda m: {"hidden_size": 4, "num_hidden_layers": 2},
        raising=True,
    )

    async def _prof(model_id, emb, maxb, batches):
        return {}

    monkeypatch.setattr(cm, "profile_cluster", _prof, raising=True)

    with TestClient(srv.app) as client:
        req = {"model": "m", "kv_bits": "8bit", "seq_len": 32, "max_batch_exp": 1}
        r = client.post("/v1/prepare_topology", json=req)
        assert r.status_code == 500


def test_prepare_topology_auto_success(monkeypatch):
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = _create_server(cm, im, mm)

    monkeypatch.setattr(
        "dnet.api.http_api.get_model_config_json",
        lambda m: {"hidden_size": 8, "num_hidden_layers": 4},
        raising=True,
    )

    async def _prof(model_id, emb, maxb, batches):
        return {"S": object()}

    monkeypatch.setattr(cm, "profile_cluster", _prof, raising=True)

    from tests.fakes import FakeModelProfile as _MP

    monkeypatch.setattr(
        "dnet.api.http_api.profile_model",
        lambda repo_id, batch_sizes, sequence_length: _MP(),
        raising=True,
    )

    async def _solve(profiles, model_profile, model_name, num_layers, kv_bits):
        dev = DnetDeviceProperties(
            is_manager=False,
            is_busy=False,
            instance="S1",
            server_port=8001,
            shard_port=9001,
            local_ip="10.0.0.1",
        )
        return TopologyInfo(
            model=model_name,
            kv_bits=kv_bits,
            num_layers=int(num_layers),
            devices=[dev],
            assignments=[],
            solution=None,
        )

    monkeypatch.setattr(cm, "solve_topology", _solve, raising=True)

    with TestClient(srv.app) as client:
        req = {"model": "m", "kv_bits": "8bit", "seq_len": 32, "max_batch_exp": 1}
        r = client.post("/v1/prepare_topology", json=req)
        assert r.status_code == 200
        topo = r.json()
        assert topo["model"] == "m" and topo["num_layers"] == 4
        assert cm.current_topology is not None


def test_load_model_bootstrap_no_topology_no_model_returns_failure():
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        r = client.post("/v1/load_model", json={"kv_bits": "8bit"})
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is False
        assert "No topology configured" in body.get("message", "")


def test_load_model_bootstrap_profiles_empty_returns_failure(monkeypatch):
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = _create_server(cm, im, mm)

    monkeypatch.setattr(
        "dnet.api.http_api.get_model_config_json",
        lambda m: {"hidden_size": 8, "num_hidden_layers": 4},
        raising=True,
    )

    async def _prof(model_id, emb, maxb, batches):
        return {}

    monkeypatch.setattr(cm, "profile_cluster", _prof, raising=True)

    with TestClient(srv.app) as client:
        r = client.post(
            "/v1/load_model",
            json={"model": "m", "kv_bits": "8bit", "seq_len": 64, "batch_size": 1},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is False
        assert "No profiles collected" in body.get("message", "")


def test_load_model_bootstrap_success_connects(monkeypatch):
    cm = FakeClusterManager({})
    im = FakeInferenceManager(grpc_port=55555)
    mm = FakeModelManager(current_model_id=None, load_success=True)
    srv = _create_server(cm, im, mm)

    monkeypatch.setattr(
        "dnet.api.http_api.get_model_config_json",
        lambda m: {"hidden_size": 16, "num_hidden_layers": 6},
        raising=True,
    )

    async def _prof(model_id, emb, maxb, batches):
        return {"S": object()}

    monkeypatch.setattr(cm, "profile_cluster", _prof, raising=True)

    from tests.fakes import FakeModelProfile as _MP2

    monkeypatch.setattr(
        "dnet.api.http_api.profile_model",
        lambda repo_id, batch_sizes, sequence_length: _MP2(),
        raising=True,
    )

    async def _solve(profiles, model_profile, model_name, num_layers, kv_bits):
        dev = DnetDeviceProperties(
            is_manager=False,
            is_busy=False,
            instance="S1",
            server_port=8001,
            shard_port=9001,
            local_ip="10.0.0.1",
        )
        return TopologyInfo(
            model=model_name,
            kv_bits=kv_bits,
            num_layers=int(num_layers),
            devices=[dev],
            assignments=[
                LayerAssignment(
                    instance="S1",
                    layers=[[0]],
                    next_instance=None,
                    window_size=1,
                    residency_size=1,
                )
            ],
            solution=None,
        )

    monkeypatch.setattr(cm, "solve_topology", _solve, raising=True)

    with TestClient(srv.app) as client:
        r = client.post(
            "/v1/load_model",
            json={"model": "m", "kv_bits": "8bit", "seq_len": 64, "batch_size": 1},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert (
            im.connected is not None
            and im.connected[0] == "10.0.0.1"
            and im.connected[1] == 9001
        )


def test_load_model_existing_topology_model_manager_exception(monkeypatch):
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = _create_server(cm, im, mm)
    dev = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="S1",
        server_port=8001,
        shard_port=9001,
        local_ip="10.0.0.1",
    )
    cm.current_topology = TopologyInfo(
        model="m",
        kv_bits="8bit",
        num_layers=2,
        devices=[dev],
        assignments=[
            LayerAssignment(
                instance="S1",
                layers=[[0, 1]],
                next_instance=None,
                window_size=1,
                residency_size=1,
            )
        ],
        solution=None,
    )

    async def _raise(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(mm, "load_model", _raise, raising=True)

    with TestClient(srv.app) as client:
        r = client.post("/v1/load_model", json={"model": None, "kv_bits": "8bit"})
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is False
        assert "boom" in (body.get("message") or "")


# NOTE: this looks insane, maybe replace?
@pytest.mark.parametrize(
    "devices,assignments",
    [
        (
            [
                {
                    "instance": "X",
                    "local_ip": "1.2.3.4",
                    "server_port": 8,
                    "shard_port": 9,
                },
                {
                    "instance": "X",
                    "local_ip": "1.2.3.5",
                    "server_port": 10,
                    "shard_port": 11,
                },
            ],
            [
                {
                    "instance": "X",
                    "layers": [[0]],
                    "next_instance": None,
                    "window_size": 1,
                    "residency_size": 1,
                }
            ],
        ),
        (
            [
                {
                    "instance": "A",
                    "local_ip": "1.2.3.4",
                    "server_port": 8,
                    "shard_port": 9,
                },
            ],
            [
                {
                    "instance": "B",
                    "layers": [[0]],
                    "next_instance": None,
                    "window_size": 1,
                    "residency_size": 1,
                }
            ],
        ),
    ],
)
def test_prepare_topology_manual_validation_errors(devices, assignments):
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        r = client.post(
            "/v1/prepare_topology_manual",
            json={
                "model": "m",
                "kv_bits": "8bit",
                "devices": devices,
                "assignments": assignments,
                "num_layers": 1,
            },
        )
        # NOTE: implementation wraps HTTPException in a 500 in some error paths
        assert r.status_code in (400, 500)
        body = r.json()
        assert isinstance(body, dict) and "detail" in body


def test_prepare_topology_manual_auto_ring_fill():
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        devices = [
            {"instance": "A", "local_ip": "1.2.3.4", "server_port": 8, "shard_port": 9},
            {
                "instance": "B",
                "local_ip": "1.2.3.5",
                "server_port": 10,
                "shard_port": 11,
            },
        ]
        # provide no next_instance on either; order sorted by min layer index becomes ring A->B->A
        assignments = [
            {
                "instance": "B",
                "layers": [[2, 3]],
                "next_instance": None,
                "window_size": 1,
                "residency_size": 1,
            },
            {
                "instance": "A",
                "layers": [[0, 1]],
                "next_instance": None,
                "window_size": 1,
                "residency_size": 1,
            },
        ]
        r = client.post(
            "/v1/prepare_topology_manual",
            json={
                "model": "m",
                "kv_bits": "8bit",
                "devices": devices,
                "assignments": assignments,
                "num_layers": 4,
            },
        )
        assert r.status_code == 200
        topo = r.json()
        amap = {a["instance"]: a["next_instance"] for a in topo["assignments"]}
        assert amap["A"] == "B" and amap["B"] == "A"


def test_prepare_topology_manual_infers_num_layers_when_missing():
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        devices = [
            {"instance": "A", "local_ip": "1.2.3.4", "server_port": 8, "shard_port": 9},
            {
                "instance": "B",
                "local_ip": "1.2.3.5",
                "server_port": 10,
                "shard_port": 11,
            },
        ]
        assignments = [
            {
                "instance": "A",
                "layers": [[0, 2, 5]],
                "next_instance": None,
                "window_size": 1,
                "residency_size": 1,
            },
            {
                "instance": "B",
                "layers": [[1, 3]],
                "next_instance": None,
                "window_size": 1,
                "residency_size": 1,
            },
        ]
        r = client.post(
            "/v1/prepare_topology_manual",
            json={
                "model": "m",
                "kv_bits": "8bit",
                "devices": devices,
                "assignments": assignments,
            },
        )
        assert r.status_code == 200
        topo = r.json()
        assert topo["num_layers"] == 6


def test_httpserver_lifecycle_start_shutdown(patch_api_hypercorn_serve):
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    srv = HTTPServer(0, cm, im, mm, node_id="node-A")

    async def main():
        stop = asyncio.Future()
        await srv.start(shutdown_trigger=lambda: stop)
        for _ in range(50):  # give the background serve task a moment to set the flag
            if patch_api_hypercorn_serve["ok"]:
                break
            await asyncio.sleep(0.01)
        assert srv.http_server is not None and patch_api_hypercorn_serve["ok"] is True
        stop.set_result(None)
        ok = await srv.wait_closed(timeout=0.2)
        assert ok is True

    asyncio.run(main())


def test_chat_completions_streaming_done_once_and_last_line():
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id="m")
    srv = _create_server(cm, im, mm)
    with TestClient(srv.app) as client:
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        with client.stream("POST", "/v1/chat/completions", json=payload) as r:
            assert r.status_code == 200
            lines = [ln for ln in r.iter_lines() if ln]
            done_count = sum(1 for ln in lines if ln.strip() == "data: [DONE]")
            assert done_count == 1
            assert lines[-1].strip() == "data: [DONE]"
