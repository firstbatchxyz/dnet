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
from ....protos.dnet_ring_pb2 import ActivationRequest

class Shard:
    def __init__(self, adapter: TopologyAdapter):
        self.node_id = 0
        self.adapter = adapter
        self.runtime: ShardRuntime = adapter.runtime

    async def start(self):
        loop = asyncio.get_running_loop()
        self.runtime.attach_loop(loop)
        self.runtime.start()  # starts compute_thread
        await self.adapter.start()

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

    async def load_model(self, req):
        self.runtime.load_model_core(req)
        await self.adapter.configure_for_model(req)

    async def unload_model(self): ...
    def queue_size(self) -> int: return self.runtime.queue_size()
