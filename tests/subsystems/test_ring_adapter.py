"""Tests: RingAdapter workers, routing, streaming, and error paths."""

import asyncio
import pytest

from dnet_p2p import DnetDeviceProperties
from dnet.shard.adapters.ring import RingAdapter
from dnet.shard.models import ShardLoadModelRequest
from dnet.config import TransportSettings
from dnet.protos.dnet_ring_pb2 import Activation, ActivationRequest
from dnet.core.types.messages import ActivationMessage

from tests.fakes import (
    FakeDiscovery,
    FakeRuntimeForAdapter,
    FakeChannel,
    FakeRingStub,
    FakeApiStub,
    FakeTBConn,
)

pytestmark = [pytest.mark.shard, pytest.mark.ring]


def _create_adapter(assigned_next=None, streaming=True):
    rt = FakeRuntimeForAdapter(assigned_next=(assigned_next or set()))
    disc = FakeDiscovery({})
    cfg = TransportSettings(streaming=streaming)
    ad = RingAdapter(runtime=rt, discovery=disc, transport_settings=cfg)
    return ad, rt


def test_create_workers_and_shutdown(monkeypatch):
    ad, rt = _create_adapter()

    async def fast_cleanup():  # stub out stream sweeper to reduce runtime
        return 0

    monkeypatch.setattr(ad._streams, "cleanup_idle_streams", fast_cleanup, raising=True)

    async def main():
        await ad.start()
        assert ad.running is True and len(ad._tasks) >= 4
        await ad.shutdown()
        assert ad.running is False

    asyncio.run(main())


def test_configure_topology_connects_next_node(patch_ring_grpc_client_ok):
    ad, rt = _create_adapter()
    patch_ring_grpc_client_ok({"addr": None})
    req_node = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="S2",
        server_port=8002,
        shard_port=9002,
        local_ip="10.0.0.2",
    )

    async def main():
        await ad.configure_topology(
            ShardLoadModelRequest(
                model_path="m",
                total_layers=2,
                layers=[0],
                warmup=False,
                next_node=req_node,
                window_size=1,
                residency_size=1,
                kv_bits="8bit",
                api_callback_address="",
            )
        )
        assert ad.next_node_stub is not None and ad.next_node_channel is not None
        assert isinstance(
            ad.next_node_channel, FakeChannel
        ) and ad.next_node_channel.addr.endswith(":9002")

    asyncio.run(main())


def test_configure_topology_with_no_next_node():
    ad, rt = _create_adapter()

    async def main():
        await ad.configure_topology(
            ShardLoadModelRequest(
                model_path="m",
                total_layers=2,
                layers=[0],
                warmup=False,
                next_node=None,
                window_size=1,
                residency_size=1,
                kv_bits="8bit",
                api_callback_address="",
            )
        )
        assert ad.next_node is None and ad.next_node_stub is None

    asyncio.run(main())


def test_connect_with_tb_when_available(monkeypatch, patch_ring_grpc_client_ok):
    ad, rt = _create_adapter()
    patch_ring_grpc_client_ok({"addr": None})

    monkeypatch.setattr(
        "dnet.shard.adapters.ring.discover_thunderbolt_connection",
        lambda a, b: FakeTBConn("192.168.1.50"),
        raising=True,
    )
    ad.next_node = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="S2",
        server_port=8002,
        shard_port=9002,
        local_ip="10.0.0.2",
    )

    async def main():
        ok = await ad._connect_next_node()
        assert ok is True and isinstance(ad.next_node_channel, FakeChannel)
        assert ad.next_node_channel.addr.startswith("192.168.1.50:")

    asyncio.run(main())


def test_reset_topology_closes_channels(monkeypatch):
    ad, rt = _create_adapter()
    ch1 = FakeChannel("10.0.0.2:9002")
    ad.next_node_channel = ch1
    ch2 = FakeChannel("127.0.0.1:5050")
    ad.api_channel = ch2
    ad.api_stub = FakeApiStub(ch2)
    ad.api_address = "127.0.0.1:5050"

    async def main():
        await ad.reset_topology()
        assert ad.next_node_channel is None and ad.next_node_stub is None
        assert ch1.closed is True
        assert ad.api_channel is None and ad.api_stub is None and ad.api_address is None
        assert ch2.closed is True

    asyncio.run(main())


def test_ingress_worker_local_layer_deserializes(monkeypatch, wait_until):
    from dnet.core.types.messages import ActivationMessage

    ad, rt = _create_adapter(assigned_next={1})
    fake_msg = ActivationMessage(
        nonce="n",
        pool_id=0,
        batch_size=1,
        shape=(1,),
        dtype="float32",
        layer_id=0,
        timestamp=0,
        node_origin="S",
        callback_url="cb",
    )
    monkeypatch.setattr(
        ad.codec, "deserialize", lambda req: fake_msg, raising=True
    )  # to avoid pool work

    async def main():
        await ad.start()
        act = Activation(
            data=b"\x00\x00\x80?", batch_size=1, shape=[1], dtype="float32", layer_id=0
        )
        req = ActivationRequest(
            nonce="n", activation=act, timestamp=0, node_origin="S", callback_url="cb"
        )
        await ad.ingress_q.put(req)
        ok = await wait_until(lambda: not rt.activation_recv_queue.empty(), timeout=0.5)
        assert ok is True  # placed into runtime.activation_recv_queue
        await ad.shutdown()

    asyncio.run(main())


def test_ingress_worker_forwards_nonlocal(patch_ring_grpc_client_ok, wait_until):
    ad, rt = _create_adapter(assigned_next=set())
    patch_ring_grpc_client_ok({"addr": None})
    ad.next_node = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="S2",
        server_port=8002,
        shard_port=9002,
        local_ip="10.0.0.2",
    )

    async def main():
        await ad.start()
        act = Activation(
            data=b"\x00\x00\x80?", batch_size=1, shape=[1], dtype="float32", layer_id=0
        )
        req = ActivationRequest(
            nonce="n2", activation=act, timestamp=0, node_origin="S", callback_url="cb"
        )
        await ad.ingress_q.put(req)
        # Wait for stream creation and first frame
        ok = await wait_until(
            lambda: (
                ad._streams.get_ctx("n2") is not None
                and ad._streams.get_ctx("n2").last_seq >= 1
            ),
            timeout=0.5,
        )
        assert ok is True  # one frame should have been enqueued
        ctx = ad._streams.get_ctx("n2")
        frame = await ctx.queue.get()
        assert frame.request.nonce == "n2"
        await ad.shutdown()

    asyncio.run(main())


def test_egress_worker_routes_to_correct_queues(monkeypatch, wait_until):
    ad, rt = _create_adapter()
    calls = {"act": 0, "tok": 0}

    async def fake_send_activation(msg):
        calls["act"] += 1

    async def fake_send_token(msg):
        calls["tok"] += 1

    monkeypatch.setattr(ad, "_send_activation", fake_send_activation, raising=True)
    monkeypatch.setattr(ad, "_send_token", fake_send_token, raising=True)

    async def main():
        await ad.start()
        m1 = ActivationMessage(  # non-final
            nonce="a",
            pool_id=0,
            batch_size=1,
            shape=(1,),
            dtype="float32",
            layer_id=0,
            timestamp=0,
            node_origin="S",
            callback_url="",
        )
        m1.is_final = False
        rt.activation_send_queue.put(m1)
        m2 = ActivationMessage(  # final
            nonce="b",
            pool_id=0,
            batch_size=1,
            shape=(1,),
            dtype="float32",
            layer_id=0,
            timestamp=0,
            node_origin="S",
            callback_url="",
        )
        m2.is_final = True
        rt.activation_send_queue.put(m2)
        ok = await wait_until(
            lambda: calls["act"] == 1 and calls["tok"] == 1, timeout=0.5
        )
        assert ok is True
        await ad.shutdown()

    asyncio.run(main())


def test_send_activation_serializes_and_enqueues(patch_ring_grpc_client_ok):
    ad, rt = _create_adapter()
    patch_ring_grpc_client_ok({"addr": None})
    ad.next_node = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="S2",
        server_port=8002,
        shard_port=9002,
        local_ip="10.0.0.2",
    )

    async def main():
        from mlx.core import array, float32

        await ad._connect_next_node()
        ctx = await ad._streams.get_or_create_stream(  # create stream
            "x", ad.next_node_stub.StreamActivations
        )
        msg = ActivationMessage(
            nonce="x",
            pool_id=0,
            batch_size=1,
            shape=(1,),
            dtype="float32",
            layer_id=0,
            timestamp=0,
            node_origin="S",
            callback_url="",
        )
        msg.tensor = array([1.0], dtype=float32)
        await ad._send_activation(msg)
        assert (await ctx.queue.get()) is not None
        assert msg.tensor is None and msg.dtype == rt._wire_dtype_str

    asyncio.run(main())


def test_send_token_uses_callback_and_builds_stub(monkeypatch):
    ad, rt = _create_adapter()
    monkeypatch.setattr(
        "dnet.shard.adapters.ring.aio_grpc.insecure_channel",
        lambda addr, options=None: FakeChannel(addr),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.adapters.ring.shard_api_comm_pb2_grpc.ShardApiServiceStub",
        lambda ch: FakeApiStub(ch),
        raising=True,
    )

    async def main():
        msg = ActivationMessage(
            nonce="t1",
            pool_id=0,
            batch_size=1,
            shape=(1,),
            dtype="float32",
            layer_id=0,
            timestamp=0,
            node_origin="S",
            callback_url="grpc://127.0.0.1:5050",  # valid grpc:// URL should be parsed
        )
        msg.is_final = True
        msg.token_id = 7
        await ad._send_token(msg)
        assert (
            ad.api_address == "127.0.0.1:5050"
        )  # fallback to api_callback_address when no callback_url
        msg2 = ActivationMessage(
            nonce="t2",
            pool_id=0,
            batch_size=1,
            shape=(1,),
            dtype="float32",
            layer_id=0,
            timestamp=0,
            node_origin="S",
            callback_url="",
        )
        msg2.is_final = True
        msg2.token_id = 8
        ad.api_callback_address = "10.0.0.9:5055"
        await ad._send_token(msg2)
        assert ad.api_address == "10.0.0.9:5055"

    asyncio.run(main())


def test_admit_frame_drops_when_not_running():
    ad, rt = _create_adapter()  # do not call start(); running remains False
    act = Activation(data=b"", batch_size=1, shape=[1], dtype="float32", layer_id=0)
    req = ActivationRequest(
        nonce="z", activation=act, timestamp=0, node_origin="S", callback_url="cb"
    )

    async def main():
        await ad.admit_frame(req)
        assert ad.ingress_q.empty()

    asyncio.run(main())


def test_forward_activation_stream_disabled(monkeypatch):
    ad, rt = _create_adapter(streaming=False)

    async def main():
        act = Activation(data=b"", batch_size=1, shape=[1], dtype="float32", layer_id=0)
        req = ActivationRequest(
            nonce="q1", activation=act, timestamp=0, node_origin="S", callback_url="cb"
        )
        await ad._forward_activation(req)
        assert ad._streams.get_ctx("q1") is None

    asyncio.run(main())


def test_send_activation_without_stub_noop():
    ad, rt = _create_adapter()

    async def main():
        msg = ActivationMessage(
            nonce="x",
            pool_id=0,
            batch_size=1,
            shape=(1,),
            dtype="float32",
            layer_id=0,
            timestamp=0,
            node_origin="S",
            callback_url="",
        )
        await ad._send_activation(msg)  # should not raise
        # No stream should be created since there's no next node stub
        assert ad._streams.get_ctx("x") is None

    asyncio.run(main())


def test_send_token_invalid_callback_url_and_no_fallback(monkeypatch):
    ad, rt = _create_adapter()

    async def main():
        msg = ActivationMessage(
            nonce="b1",
            pool_id=0,
            batch_size=1,
            shape=(1,),
            dtype="float32",
            layer_id=0,
            timestamp=0,
            node_origin="S",
            callback_url="http://bad",
        )
        msg.is_final = True
        await ad._send_token(msg)
        assert ad.api_address is None

    asyncio.run(main())


def test_send_token_channel_creation_failure(monkeypatch):
    ad, rt = _create_adapter()

    def boom(addr, options=None):  # force channel creation to raise
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "dnet.shard.adapters.ring.aio_grpc.insecure_channel", boom, raising=True
    )

    async def main():
        msg = ActivationMessage(
            nonce="b2",
            pool_id=0,
            batch_size=1,
            shape=(1,),
            dtype="float32",
            layer_id=0,
            timestamp=0,
            node_origin="S",
            callback_url="",
        )
        msg.is_final = True
        ad.api_callback_address = "10.0.0.1:5050"
        await ad._send_token(msg)
        assert ad.api_channel is None  # creation failed, no channel

    asyncio.run(main())


def test_ingress_deserialize_exception_is_handled(monkeypatch, wait_until):
    ad, rt = _create_adapter(assigned_next={1})

    def boom(req):
        raise RuntimeError("decode")

    monkeypatch.setattr(ad.codec, "deserialize", boom, raising=True)

    async def main():
        await ad.start()
        act = Activation(
            data=b"\x00\x00\x80?", batch_size=1, shape=[1], dtype="float32", layer_id=0
        )
        req = ActivationRequest(
            nonce="n3", activation=act, timestamp=0, node_origin="S", callback_url="cb"
        )
        await ad.ingress_q.put(req)
        # Wait until ingress_q is drained, then ensure nothing enqueued to runtime
        ok = await wait_until(lambda: ad.ingress_q.empty(), timeout=0.5)
        assert ok is True and rt.activation_recv_queue.empty()
        await ad.shutdown()

    asyncio.run(main())


def test_reconnect_next_node_replaces_channel(patch_ring_grpc_client_ok):
    ad, rt = _create_adapter()
    ch_old = FakeChannel("10.0.0.2:9002")
    ad.next_node_channel = ch_old
    ad.next_node_stub = FakeRingStub(ch_old)
    patch_ring_grpc_client_ok({"addr": None})
    ad.next_node = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="S2",
        server_port=8002,
        shard_port=9002,
        local_ip="10.0.0.2",
    )

    async def main():
        ok = await ad._reconnect_next_node()
        assert (
            ok is True
            and ch_old.closed is True
            and isinstance(ad.next_node_channel, FakeChannel)
        )

    asyncio.run(main())
