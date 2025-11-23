"""Tests: Shard HTTP server (health, load/unload, cleanup, profile, latency, lifecycle)."""

import asyncio
import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeChannel, FakeRingStub, FakeShard, FakeDiscovery

pytestmark = [pytest.mark.shard, pytest.mark.http]


def test_health_endpoint_basic(shard_http_server):
    srv, shard = shard_http_server
    with TestClient(srv.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and body["grpc_port"] == 9001
        assert body["instance"] == "S1"


def test_load_model_route_success(shard_http_server):
    srv, shard = shard_http_server
    with TestClient(srv.app) as client:
        payload = {
            "model_path": "m",
            "total_layers": 1,
            "layers": [0],
            "warmup": False,
            "next_node": None,
            "window_size": 1,
            "residency_size": 1,
            "kv_bits": "8bit",
            "api_callback_address": "cb",
        }
        r1 = client.post("/load_model", json=payload)
        assert r1.status_code == 200 and r1.json()["success"] is True


def test_unload_model_route_success(shard_http_server):
    srv, shard = shard_http_server
    with TestClient(srv.app) as client:
        r2 = client.post("/unload_model")
        assert r2.status_code == 200 and r2.json()["success"] is True


def test_cleanup_repacked_calls_helper(monkeypatch, shard_http_server):
    srv, shard = shard_http_server
    called = {"ok": False}

    def _del(**kwargs):
        called["ok"] = True
        return []

    monkeypatch.setattr(
        "dnet.shard.http_api.delete_repacked_layers", _del, raising=True
    )
    with TestClient(srv.app) as client:
        r = client.post("/cleanup_repacked", json={"model_id": "m", "all": False})
        assert r.status_code == 200 and called["ok"] is True


def test_profile_route(monkeypatch, shard_http_server):
    srv, shard = shard_http_server
    monkeypatch.setattr(
        "dnet.shard.http_api.profile_device_via_subprocess",
        lambda repo_id, max_batch_exp, debug: {"ok": True},
        raising=True,
    )
    with TestClient(srv.app) as client:
        r = client.post(
            "/profile",
            json={
                "repo_id": "m",
                "max_batch_exp": 1,
                "thunderbolts": {},
                "payload_sizes": [],
                "devices": {},
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "profile" in body and isinstance(body["profile"], dict)


def test_profile_route_error_returns_500(monkeypatch, shard_http_server):
    srv, shard = shard_http_server

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "dnet.shard.http_api.profile_device_via_subprocess", _boom, raising=True
    )
    with TestClient(srv.app, raise_server_exceptions=False) as client:
        r = client.post(
            "/profile",
            json={
                "repo_id": "m",
                "max_batch_exp": 1,
                "thunderbolts": {},
                "payload_sizes": [],
                "devices": {},
            },
        )
        assert r.status_code == 500


def test_measure_latency_skips_self_and_managers(monkeypatch, shard_http_server):
    srv, shard = shard_http_server
    devices = {  # one is self and another is manager -> both should be skipped
        "S1": {
            "instance": "S1",
            "local_ip": "127.0.0.1",
            "server_port": 8000,
            "shard_port": 9001,
            "is_manager": False,
            "is_busy": False,
        },
        "M": {
            "instance": "M",
            "local_ip": "127.0.0.1",
            "server_port": 8001,
            "shard_port": 9002,
            "is_manager": True,
            "is_busy": False,
        },
    }

    # ensure we never attempt to connect to any device when all are skipped
    called = {"count": 0}

    def _should_not_be_called(*a, **k):
        called["count"] += 1
        raise AssertionError(
            "insecure_channel should not be called for skipped devices"
        )

    monkeypatch.setattr(
        "dnet.shard.http_api.aio_grpc.insecure_channel",
        _should_not_be_called,
        raising=True,
    )

    with TestClient(srv.app) as client:
        r = client.post(
            "/measure_latency",
            json={"devices": devices, "thunderbolts": {}, "payload_sizes": [1, 2]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["latency"]["results"] == {}
        assert called["count"] == 0


@pytest.mark.parametrize(
    "ml_success,use_tb,expected_addr",
    [
        (True, False, "10.0.0.2:9102"),
        (True, True, "192.168.2.100:9102"),
        (False, False, "10.0.0.2:9102"),
    ],
)
def test_measure_latency_remote_matrix(
    patch_http_grpc_client, shard_http_server, ml_success, use_tb, expected_addr
):
    srv, shard = shard_http_server
    FakeRingStub.ml_success = bool(ml_success)
    FakeRingStub.ml_node_id = "S2"
    FakeRingStub.ml_message = "ok" if ml_success else "rpc-fail"

    seen = patch_http_grpc_client({"addr": None})

    devices = {
        "S2": {
            "instance": "S2",
            "local_ip": "10.0.0.2",
            "server_port": 8002,
            "shard_port": 9102,
            "is_manager": False,
            "is_busy": False,
        }
    }
    thunderbolts = (
        {
            "S2": {
                "ip_addr": "192.168.2.100",
                "instance": {"uuid": "u1", "name": "tb", "device": "dev"},
            }
        }
        if use_tb
        else {}
    )
    with TestClient(srv.app) as client:
        r = client.post(
            "/measure_latency",
            json={
                "devices": devices,
                "thunderbolts": thunderbolts,
                "payload_sizes": [1, 8],
            },
        )
        assert r.status_code == 200
        assert seen["addr"] == expected_addr
        body = r.json()
        res = body["latency"]["results"]["S2"]
        ms = res["measurements"]
        if ml_success:
            assert (
                isinstance(ms, list) and len(ms) == 2 and all(m["success"] for m in ms)
            )
        else:
            assert any(not m["success"] for m in ms)


def test_measure_latency_failure_propagates_in_results(monkeypatch, shard_http_server):
    srv, shard = shard_http_server
    FakeRingStub.ml_success = False
    FakeRingStub.ml_message = "rpc-fail"

    monkeypatch.setattr(
        "dnet.shard.http_api.aio_grpc.insecure_channel",
        lambda a: FakeChannel(a),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.http_api.DnetRingServiceStub",
        lambda ch: FakeRingStub(ch),
        raising=True,
    )

    devices = {
        "S2": {
            "instance": "S2",
            "local_ip": "10.0.0.2",
            "server_port": 8002,
            "shard_port": 9102,
            "is_manager": False,
        }
    }
    with TestClient(srv.app) as client:
        r = client.post(
            "/measure_latency",
            json={"devices": devices, "thunderbolts": {}, "payload_sizes": [1]},
        )
        assert r.status_code == 200
        body = r.json()
        res = body["latency"]["results"]["S2"]
        assert res["success"] is True  # call completed
        assert any(
            not m["success"] for m in res["measurements"]
        )  # but measurement failed

    # remaining failure path exercised in the matrix above (ml_success=False)


def test_load_model_route_error_returns_failure(monkeypatch, shard_http_server):
    srv, shard = shard_http_server

    async def _boom(req):
        raise RuntimeError("load-fail")

    monkeypatch.setattr(srv.shard, "load_model", _boom, raising=True)
    with TestClient(srv.app) as client:
        payload = {
            "model_path": "m",
            "total_layers": 1,
            "layers": [0],
            "warmup": False,
            "next_node": None,
            "window_size": 1,
            "residency_size": 1,
            "kv_bits": "8bit",
            "api_callback_address": "cb",
        }
        r = client.post("/load_model", json=payload)
        assert r.status_code == 200 and r.json()["success"] is False


def test_unload_model_route_error_returns_failure(monkeypatch, shard_http_server):
    srv, shard = shard_http_server

    async def _boom():
        raise RuntimeError("unload-fail")

    monkeypatch.setattr(srv.shard, "unload_model", _boom, raising=True)
    with TestClient(srv.app) as client:
        r = client.post("/unload_model")
        assert r.status_code == 200 and r.json()["success"] is False


def test_cleanup_repacked_error_returns_500(monkeypatch, shard_http_server):
    srv, shard = shard_http_server

    def _boom(**kwargs):
        raise RuntimeError("cleanup-fail")

    monkeypatch.setattr(
        "dnet.shard.http_api.delete_repacked_layers", _boom, raising=True
    )
    with TestClient(srv.app) as client:
        r = client.post("/cleanup_repacked", json={"model_id": "m", "all": False})
        assert r.status_code == 500 and "error" in r.json()


def test_httpserver_lifecycle_start_shutdown(patch_shard_hypercorn_serve):
    # instantiate without pre-binding routes to avoid double registration
    shard = FakeShard(1)
    disc = FakeDiscovery({})
    setattr(disc, "instance_name", lambda: "S1")
    from dnet.shard.http_api import HTTPServer

    srv = HTTPServer(http_port=0, grpc_port=9001, shard=shard, discovery=disc)

    async def main():
        stop = asyncio.Future()
        await srv.start(shutdown_trigger=lambda: stop)
        for _ in range(50):
            if patch_shard_hypercorn_serve["ok"]:
                break
            await asyncio.sleep(0.01)
        assert srv.http_server is not None and patch_shard_hypercorn_serve["ok"] is True
        stop.set_result(None)
        ok = await srv.wait_closed(timeout=0.2)
        assert ok is True

    asyncio.run(main())
