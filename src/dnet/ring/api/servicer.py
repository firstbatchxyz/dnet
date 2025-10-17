"""gRPC servicer for API node (receives callbacks from shards)."""

from typing import TYPE_CHECKING

import grpc

from ...protos import shard_api_comm_pb2 as pb2
from ...protos import shard_api_comm_pb2_grpc as pb2_grpc
from ...utils.logger import logger

if TYPE_CHECKING:
    pass


class ShardApiServicer(pb2_grpc.ShardApiServiceServicer):
    """gRPC servicer for shard -> API callbacks."""

    def __init__(self, api_node):
        # api_node: RingApiNode
        self.api_node = api_node

    async def SendToken(
        self, request: pb2.TokenRequest, context: grpc.aio.ServicerContext
    ):  # type: ignore[override]
        try:
            nonce = request.nonce
            token_id = int(request.token_id)
            future = self.api_node.pending_requests.get(nonce)
            if future is None:
                msg = f"Nonce {nonce} not found in pending requests"
                logger.warning(msg)
                return pb2.TokenResponse(success=False, message=msg)
            if future.done():
                msg = f"Nonce {nonce} already resolved"
                logger.warning(msg)
                return pb2.TokenResponse(success=False, message=msg)
            # Resolve future with token id (head_on_shard path)
            future.set_result(token_id)
            return pb2.TokenResponse(success=True, message="Token received")
        except Exception as e:
            logger.exception("Error handling SendToken: %s", e)
            return pb2.TokenResponse(success=False, message=str(e))
