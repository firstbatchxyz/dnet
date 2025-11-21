"""
Topology/Strategy agnostic shard

Startup
build runtime = ShardRuntime(compute_config)
build adapter = RingAdapter(runtime, transport_config, topology_config)
build shard = RingShard(runtime, adapter, ports, discovery, …)
grpc_servicer = ShardServicer(shard)
app = FastAPI(); attach_routes(app, shard)
"""

import asyncio
from .runtime import ShardRuntime
from .adapters.base import TopologyAdapter
from dnet.protos.dnet_ring_pb2 import ActivationRequest
from dnet.utils.banner import print_startup_banner
from .models import ShardLoadModelResponse, ShardUnloadModelResponse


from dnet.utils.repack import delete_repacked_layers


class Shard:
    def __init__(self, shard_id, adapter: TopologyAdapter):
        self.node_id = shard_id
        self.adapter = adapter
        self.runtime: ShardRuntime = adapter.runtime
        print_startup_banner(tag="shard")

    async def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.runtime.attach_loop(loop)
        self.runtime.start()  # starts compute_thread
        await self.adapter.start()

    async def shutdown(self) -> None:
        self.runtime.shutdown()
        await self.adapter.shutdown()

    async def admit_frame(self, request: ActivationRequest) -> None:
        while self.adapter.running:
            try:
                self.adapter.ingress_q.put_nowait(request)
                return
            except asyncio.QueueFull:
                await asyncio.sleep(0)
        return

    async def reset_cache(self):
        self.runtime.reset_cache()

    async def load_model(self, req) -> ShardLoadModelResponse:
        self.runtime.load_model_core(req)
        await self.adapter.configure_topology(req)
        return ShardLoadModelResponse(
            success=True,
            message="Model loaded successfully",
            layers_loaded=self.runtime.assigned_layers,
            load_time_ms=100,
        )

    async def unload_model(self) -> ShardUnloadModelResponse:
        await self.adapter.reset_topology()
        model_path = self.runtime.model_path
        response = self.runtime.unload_model_core()
        if response.success and model_path:
            delete_repacked_layers(
                current_model_path=model_path,
            )
        return response

    def queue_size(self) -> int:
        return self.runtime.queue_size()
