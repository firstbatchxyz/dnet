"""Tests: Shard wrapper start/shutdown wiring and admit_frame queueing."""

import asyncio
import pytest

from dnet.shard.shard import Shard
from dnet.shard.models import ShardUnloadModelResponse
from tests.fakes import FakeShard, FakeAdapter, FakeRuntimeMinimal

pytestmark = [pytest.mark.shard]


def test_start_shutdown_wires_runtime_and_adapter():
    s = FakeShard("S1")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def main():
        await s.start(loop)
        assert (
            s.adapter.running is True and getattr(s.runtime, "_started", False) is True
        )
        await s.shutdown()
        assert (
            s.adapter.running is False and getattr(s.runtime, "_started", True) is False
        )

    loop.run_until_complete(main())
    loop.close()


def test_admit_frame_puts_in_ingress_queue():
    s = FakeShard("S1")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def main():
        await s.adapter.start()
        from dnet.protos.dnet_ring_pb2 import Activation, ActivationRequest

        req = ActivationRequest(
            nonce="n",
            activation=Activation(
                data=b"", batch_size=1, shape=[1], dtype="float32", layer_id=0
            ),
            timestamp=0,
            node_origin="S",
            callback_url="cb",
        )
        await s.admit_frame(req)
        assert not s.adapter.ingress_q.empty()

    loop.run_until_complete(main())
    loop.close()


def test_reset_cache_calls_runtime():
    s = FakeShard("S1")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def main():
        await s.reset_cache()
        assert s._reset_called is True and s.runtime._cache_reset is True

    loop.run_until_complete(main())
    loop.close()


def test_load_and_unload_model_calls_components(monkeypatch):
    rt = FakeRuntimeMinimal("S1")
    adapter = FakeAdapter(runtime=rt)
    s = Shard(shard_id="S1", adapter=adapter)

    called = {"ok": False}

    def _del(**kwargs):
        called["ok"] = True
        return []

    monkeypatch.setattr("dnet.shard.shard.delete_repacked_layers", _del, raising=True)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def main():
        from dnet.shard.models import ShardLoadModelRequest

        req = ShardLoadModelRequest(
            model_path="m",
            total_layers=1,
            layers=[0],
            warmup=False,
            next_node=None,
            window_size=1,
            residency_size=1,
            kv_bits="8bit",
            api_callback_address="cb",
        )
        resp = await s.load_model(req)
        assert resp.success is True and s.runtime.assigned_layers == [0]
        ur = await s.unload_model()
        assert isinstance(ur, ShardUnloadModelResponse)
        assert called["ok"] is True

    loop.run_until_complete(main())
    loop.close()
