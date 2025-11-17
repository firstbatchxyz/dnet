from .servicer import GrpcServicer
from ..shard import Shard
from dnet.protos.dnet_ring_pb2_grpc import add_DnetRingServiceServicer_to_server
from grpc import aio as aio_grpc
from typing import Optional
from .....utils.logger import logger

class GrpcServer:
    def __init__(self, shard: Shard):
        self.grpc_port: int = 58080
        self.shard = shard
        self.server: Optional[aio_grpc.Server] = None
        self.servicer = GrpcServicer(self.shard)

    async def start(self):
        """
        Start gRPC server
        """
        self.server = aio_grpc.server()
        add_DnetRingServiceServicer_to_server(self.servicer, self.server)
        listen_addr = f"[::]:{self.grpc_port}"
        self.server.add_insecure_port(listen_addr)
        try:
            await self.server.start()
        except RuntimeError:
            logger.error("Couldn't start gRPC server.")
        logger.info("gRPC server started on %s", listen_addr)