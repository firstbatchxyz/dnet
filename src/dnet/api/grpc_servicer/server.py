from __future__ import annotations

from typing import Optional
from grpc import aio as aio_grpc

from dnet.utils.logger import logger
from .servicer import ShardApiServicer
from ..inference import InferenceManager
from dnet.protos.shard_api_comm_pb2_grpc import add_ShardApiServiceServicer_to_server


class GrpcServer:
    def __init__(self, grpc_port: int, inference_manager: InferenceManager) -> None:
        self.grpc_port = grpc_port
        self.inference_manager: InferenceManager = inference_manager
        self.server: Optional[aio_grpc.Server] = None
        self.servicer = ShardApiServicer(self.inference_manager)

    async def start(self) -> None:
        self.server = aio_grpc.server()
        add_ShardApiServiceServicer_to_server(self.servicer, self.server)
        listen_addr = f"[::]:{self.grpc_port}"
        self.server.add_insecure_port(listen_addr)
        await self.server.start()
        logger.info("gRPC server started on %s", listen_addr)

    async def shutdown(self) -> None:
        if self.server:
            await self.server.stop(grace=5)
            logger.info("gRPC server on port %d stopped", self.grpc_port)
