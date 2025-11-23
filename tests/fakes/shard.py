"""Shard/adapter fakes."""

from __future__ import annotations

from typing import Any


class FakeAdapter:
    """Minimal adapter with ingress/computed/token queues and lifecycle flags."""

    def __init__(self, runtime=None):
        import asyncio
        from dnet.protos.dnet_ring_pb2 import ActivationRequest
        from dnet.core.types.messages import ActivationMessage

        self._ingress_q: asyncio.Queue[ActivationRequest] = asyncio.Queue(maxsize=8)
        self._computed_q: asyncio.Queue[ActivationMessage] = asyncio.Queue(maxsize=8)
        self._token_q: asyncio.Queue[ActivationMessage] = asyncio.Queue(maxsize=8)
        self.running: bool = False
        self.runtime = runtime
        self._last_req: Any = None

    @property
    def ingress_q(self):
        return self._ingress_q

    @property
    def activation_computed_queue(self):
        return self._computed_q

    @property
    def activation_token_queue(self):
        return self._token_q

    async def start(self):
        self.running = True

    async def ingress(self):
        return

    async def egress(self):
        return

    async def configure_topology(self, req):
        self._last_req = req

    async def reset_topology(self):
        self._last_req = None

    async def shutdown(self):
        self.running = False


class FakeShard:
    """Wrapper around FakeRuntimeMinimal + FakeAdapter for simplified shard tests."""

    def __init__(self, node_id: str = "S1"):
        from .runtime import FakeRuntimeMinimal

        self.node_id = node_id
        self.runtime = FakeRuntimeMinimal(node_id=node_id)
        self.adapter = FakeAdapter()
        self._last_admit = None
        self._reset_called = False

    async def start(self, loop):
        self.runtime.attach_loop(loop)
        self.runtime.start()
        await self.adapter.start()

    async def shutdown(self):
        self.runtime.shutdown()
        await self.adapter.shutdown()

    async def admit_frame(self, request):
        self._last_admit = request
        await self.adapter.ingress_q.put(request)

    async def reset_cache(self):
        self._reset_called = True
        self.runtime.reset_cache()

    async def load_model(self, req):
        from dnet.shard.models import ShardLoadModelResponse

        self.runtime.load_model_core(req)
        await self.adapter.configure_topology(req)
        return ShardLoadModelResponse(
            success=True,
            message="ok",
            layers_loaded=self.runtime.assigned_layers,
            load_time_ms=1.0,
        )

    async def unload_model(self):
        self.adapter  # for interface symmetry
        return self.runtime.unload_model_core()

    def queue_size(self) -> int:
        return self.runtime.queue_size()
