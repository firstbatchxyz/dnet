"""Tests: API gRPC server lifecycle and live SendToken smoke."""

import asyncio
import pytest

from dnet.api.grpc_servicer.server import GrpcServer
from tests.fakes import FakeGrpcServer, FakeInferenceManager

pytestmark = [pytest.mark.api, pytest.mark.grpc]


def test_init_sets_fields():
    mgr = FakeInferenceManager()
    srv = GrpcServer(50505, mgr)
    assert srv.grpc_port == 50505
    assert srv.server is None
    assert srv.servicer.inference_manager is mgr


def test_start_and_shutdown(monkeypatch):
    added = {}
    fake = FakeGrpcServer()

    def _server_factory():
        return fake

    def _add_servicer(servicer, server):
        added["ok"] = servicer is not None and server is fake

    monkeypatch.setattr(
        "dnet.api.grpc_servicer.server.aio_grpc.server", _server_factory, raising=True
    )
    monkeypatch.setattr(
        "dnet.api.grpc_servicer.server.add_ShardApiServiceServicer_to_server",
        _add_servicer,
        raising=True,
    )

    mgr = FakeInferenceManager()
    gs = GrpcServer(60606, mgr)

    async def run():
        await gs.start()
        assert gs.server is fake
        assert fake.added == "[::]:60606"
        assert fake.started is True
        assert added.get("ok") is True
        await gs.shutdown()
        assert fake.stopped == 5

    asyncio.run(run())


@pytest.mark.e2e
def test_server_live_sendtoken():
    import socket
    from grpc import aio as aio_grpc
    from dnet.protos import shard_api_comm_pb2 as pb2
    from dnet.protos import shard_api_comm_pb2_grpc as pb2_grpc

    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    mgr = FakeInferenceManager()
    gs = GrpcServer(port, mgr)

    async def main():
        await gs.start()
        async with aio_grpc.insecure_channel(f"localhost:{port}") as ch:
            stub = pb2_grpc.ShardApiServiceStub(ch)
            req = pb2.TokenRequest(
                nonce="n1",
                token_id=7,
                timestamp=0,
                logprob=-0.1,
                top_logprobs={7: -0.1},
            )
            resp = await stub.SendToken(req)
            assert resp.success is True
        await gs.shutdown()
        assert mgr.last is not None
        (args, kwargs) = mgr.last
        assert args[0] == "n1"
        assert int(getattr(args[1], "token_id", -1)) == 7

    asyncio.run(main())
