"""gRPC servicer for API node (receives callbacks from shards)."""

import time
from typing import TYPE_CHECKING

import grpc

from ...protos import shard_api_comm_pb2 as pb2
from ...protos import shard_api_comm_pb2_grpc as pb2_grpc
from ...utils.logger import logger
from ..api_models import RecieveResultRequest

if TYPE_CHECKING:
    from .node import RingApiNode


class ApiServicer(pb2_grpc.ShardApiServiceServicer):
    """gRPC servicer for shard -> API callbacks."""

    def __init__(self, api_node: "RingApiNode") -> None:
        """Initialize API servicer.

        Args:
            api_node: The API node instance
        """
        self.api_node = api_node

    async def SendFinalActivation(
        self,
        request: pb2.FinalActivationRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.FinalActivationResponse:
        """Handle final activation from last shard.

        Args:
            request: Final activation request from shard
            context: gRPC context

        Returns:
            Final activation response
        """
        try:
            # Transport metrics (timestamps are in ms)
            coarse_transport_ms = (time.time() * 1000.0) - float(request.timestamp)

            payload_kb = len(request.data) / 1024.0 if request.data is not None else 0.0
            logger.info(
                f"[PROFILE][API-RX] nonce={request.nonce} "
                f"payload_kb={payload_kb:.1f} "
                f"transport_coarse_ms={coarse_transport_ms}"
            )

            nonce = request.nonce
            future = self.api_node.pending_requests.get(nonce)

            if future is None:
                msg = f"Nonce {nonce} not found in pending requests"
                logger.warning(msg)
                return pb2.FinalActivationResponse(
                    success=False, message=msg, token_id=-1
                )

            if future.done():
                msg = f"Nonce {nonce} already resolved"
                logger.warning(msg)
                return pb2.FinalActivationResponse(
                    success=False, message=msg, token_id=-1
                )

            # Build the same payload the HTTP route used to provide
            recv = RecieveResultRequest(
                nonce=request.nonce,
                batch_size=request.batch_size,
                shape=tuple(request.shape),
                dtype=request.dtype,
                layer_id=request.layer_id,
                timestamp=request.timestamp,
                node_origin=request.node_origin,
                data=RecieveResultRequest.encode(bytes(request.data)),
            )

            future.set_result(recv)
            return pb2.FinalActivationResponse(
                success=True,
                message="Final activation received",
                token_id=-1,  # Token is computed by API after this callback
            )
        except Exception as e:
            logger.exception(f"Error handling SendFinalActivation: {e}")
            return pb2.FinalActivationResponse(
                success=False, message=str(e), token_id=-1
            )

    async def SendToken(
        self, request: pb2.TokenRequest, context: grpc.aio.ServicerContext
    ) -> pb2.TokenResponse:
        """Handle token send (not implemented).

        Args:
            request: Token request
            context: gRPC context

        Returns:
            Token response (unimplemented)
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("SendToken not implemented")
        return pb2.TokenResponse(success=False, message="Unimplemented")
