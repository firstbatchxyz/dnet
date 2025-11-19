import asyncio
import uuid
import grpc.aio
from typing import Optional, Any
from dnet_p2p import AsyncDnetP2P

from ....utils.logger import logger
from ....protos import shard_api_comm_pb2_grpc as pb2_grpc
from .config import ApiConfig
from .cluster import ClusterManager
from .inference import InferenceManager
from .adapters.ring import ApiRingAdapter
from .model_manager import ModelManager
from .http_api import HTTPServer
from .grpc_servicer import ShardApiServicer

class RingApiNode:
    """
    New modular Ring API Node.
    """
    def __init__(self, config: ApiConfig):
        self.config = config
        self.node_id = f"api-{uuid.uuid4().hex[:8]}"
        
        # Components
        self.discovery = AsyncDnetP2P("lib/dnet-p2p/lib")
        
        self.cluster_manager = ClusterManager(self.discovery)
        self.model_manager = ModelManager()
        self.inference_manager = InferenceManager(
            self.cluster_manager,
            self.model_manager,
            config.grpc_port,
        )
        
        self.http_server = HTTPServer(
            http_port=self.config.http_port,
            cluster_manager=self.cluster_manager,
            inference_manager=self.inference_manager,
            model_manager=self.model_manager,
            node_id=self.node_id
        )
        
        self.grpc_server: Optional[grpc.aio.Server] = None
        
    async def start(self, shutdown_trigger: Any = lambda: asyncio.Future()):
        logger.info(f"Starting RingApiNode {self.node_id}")
        
        # Start Discovery
        # Register this API node instance so shards can discover callback address
        self.discovery.create_instance(
            self.node_id,
            self.config.http_port,
            self.config.grpc_port,
            is_manager=True,
        )
        await self.discovery.async_start()
        
        # Start gRPC Server
        self.grpc_server = grpc.aio.server()
        servicer = ShardApiServicer(self.inference_manager)
        pb2_grpc.add_ShardApiServiceServicer_to_server(servicer, self.grpc_server)
        self.grpc_server.add_insecure_port(f"[::]:{self.config.grpc_port}")
        await self.grpc_server.start()
        logger.info(f"gRPC server started on port {self.config.grpc_port}")
        
        # Start HTTP Server
        await self.http_server.start(shutdown_trigger)
        logger.info(f"HTTP server started on port {self.config.http_port}")
        
    async def shutdown(self):
        logger.info("Shutting down RingApiNode")
        closed = await self.http_server.wait_closed(timeout=5.0)
        if not closed:
            await self.http_server.shutdown()
        
        if self.grpc_server:
            await self.grpc_server.stop(grace=5)
        
        await self.discovery.async_stop()
        try:
            await self.discovery.async_free_instance()
        except Exception:
            pass
        logger.info("RingApiNode shutdown complete")
