import grpc
from dnet.protos import shard_api_comm_pb2 as pb2
from dnet.protos import shard_api_comm_pb2_grpc as pb2_grpc
from dnet.utils.logger import logger
from ..inference import InferenceManager

from dnet.core.types.messages import TokenResult


class ShardApiServicer(pb2_grpc.ShardApiServiceServicer):
    """gRPC servicer for shard -> API callbacks."""

    def SendFinalActivation(self, request, context):
        pass

    def __init__(self, inference_manager: InferenceManager):
        self.inference_manager = inference_manager

    async def SendToken(
        self, request: pb2.TokenRequest, context: grpc.aio.ServicerContext
    ):
        try:
            nonce = request.nonce
            token_id = int(request.token_id)
            logprob = float(request.logprob)
            top_logprobs = dict(request.top_logprobs)

            result = TokenResult(
                token_id=token_id, logprob=logprob, top_logprobs=top_logprobs
            )

            self.inference_manager.resolve_request(nonce, result)
            return pb2.TokenResponse(success=True, message="Token received")

        except Exception as e:
            logger.exception("Error handling SendToken: %s", e)
            return pb2.TokenResponse(success=False, message=str(e))
