from .servicer import GrpcServicer
from ..shard import Shard
from dnet.protos.dnet_ring_pb2_grpc import add_DnetRingServiceServicer_to_server
from dnet.protos.dnet_cp_pb2_grpc import add_CPRingServiceServicer_to_server
from dnet.core.cp.ring_comm import CPRingServiceServicer
from grpc import aio as aio_grpc
from typing import Optional, Any, cast
from dnet.utils.logger import logger


class GrpcServer:
    def __init__(self, grpc_port: int, shard: Shard):
        self.grpc_port: int = grpc_port
        self.shard = shard
        self.server: Optional[aio_grpc.Server] = None
        self.servicer = GrpcServicer(self.shard)
        self.cp_servicer: Optional[CPRingServiceServicer] = None

    async def start(self):
        """
        Start gRPC server
        """
        self.server = aio_grpc.server()
        add_DnetRingServiceServicer_to_server(self.servicer, self.server)

        # Register CP ring service (for context parallelism block transfer)
        self.cp_servicer = CPRingServiceServicer()
        add_CPRingServiceServicer_to_server(cast(Any, self.cp_servicer), self.server)

        listen_addr = f"[::]:{self.grpc_port}"
        self.server.add_insecure_port(listen_addr)
        try:
            await self.server.start()
        except RuntimeError:
            logger.error("Couldn't start gRPC server.")
        logger.info("gRPC server started on %s", listen_addr)

    async def shutdown(self):
        """
        Shutdown gRPC server
        """
        if self.server:
            await self.server.stop(grace=5)
            logger.info("gRPC server on port %d stopped", self.grpc_port)
