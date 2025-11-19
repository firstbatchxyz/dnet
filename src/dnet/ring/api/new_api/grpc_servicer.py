import grpc
from ....protos import shard_api_comm_pb2 as pb2
from ....protos import shard_api_comm_pb2_grpc as pb2_grpc
from ....utils.logger import logger
from .inference import InferenceManager

class ShardApiServicer(pb2_grpc.ShardApiServiceServicer):
    """gRPC servicer for shard -> API callbacks."""

    def __init__(self, inference_manager: InferenceManager):
        self.inference_manager = inference_manager

    async def SendToken(
        self, request: pb2.TokenRequest, context: grpc.aio.ServicerContext
    ):  # type: ignore[override]
        try:
            nonce = request.nonce
            token_id = int(request.token_id)
            self.inference_manager.resolve_request(nonce, token_id)
            return pb2.TokenResponse(success=True, message="Token received")
        
        except Exception as e:
            logger.exception("Error handling SendToken: %s", e)
            return pb2.TokenResponse(success=False, message=str(e))
