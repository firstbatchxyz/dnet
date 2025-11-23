"""Tests: Shard gRPC server lifecycle and SendActivation smoke (pre-E2E)."""

import asyncio
import pytest
import socket

from dnet.shard.grpc_servicer.server import GrpcServer
from dnet.protos import dnet_ring_pb2 as pb2
from dnet.protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from grpc import aio as aio_grpc

from tests.fakes import FakeShard, FakeGrpcServer

pytestmark = [pytest.mark.shard, pytest.mark.grpc]


def _free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_shard_grpc_server_init_and_start_shutdown(monkeypatch):
    shard = FakeShard("S1")
    gs = GrpcServer(_free_port(), shard)

    fake = FakeGrpcServer()
    monkeypatch.setattr(
        "dnet.shard.grpc_servicer.server.aio_grpc.server", lambda: fake, raising=True
    )

    async def main():
        await gs.start()
        assert (
            gs.server is fake and fake.started is True and isinstance(fake.added, str)
        )
        await gs.shutdown()
        assert fake.stopped == 5

    asyncio.run(main())


@pytest.mark.e2e
def test_shard_grpc_server_live_send_activation():
    shard = FakeShard("S1")
    port = _free_port()
    gs = GrpcServer(port, shard)

    async def main():
        await gs.start()
        async with aio_grpc.insecure_channel(f"localhost:{port}") as ch:
            stub = DnetRingServiceStub(ch)
            req = pb2.ActivationRequest(
                nonce="n1",
                activation=pb2.Activation(
                    data=b"", batch_size=1, shape=[1], dtype="float32", layer_id=0
                ),
                timestamp=0,
                node_origin="S",
                callback_url="cb",
            )
            resp = await stub.SendActivation(req)
            assert resp.success is True and resp.node_id
        await gs.shutdown()
        assert shard._last_admit is not None

    asyncio.run(main())
