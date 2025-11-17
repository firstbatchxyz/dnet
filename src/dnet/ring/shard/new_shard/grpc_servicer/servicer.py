import grpc
import time

from .....protos.dnet_ring_pb2 import (
    ActivationRequest,
    ActivationResponse,
    HealthRequest,
    HealthResponse,
    LatencyMeasureRequest,
    LatencyMeasureResponse,
    ResetCacheRequest,
    ResetCacheResponse,
)
from .....protos.dnet_ring_pb2_grpc import DnetRingServiceServicer
from .....protos import dnet_ring_pb2 as pb2
from .....utils.logger import logger
from dnet.ring.shard.new_shard.shard import Shard

class GrpcServicer(DnetRingServiceServicer):
    """gRPC servicer implementation"""

    def __init__(self, shard: Shard):
        self.shard = shard

    async def SendActivation(
        self,
        request: ActivationRequest,
        context: grpc.aio.ServicerContext,
    ) -> ActivationResponse:
        """Handle incoming activation requests"""
        try:
            logger.debug(
                "Node %s received activation: nonce=%s, layer=%s",
                self.shard.node_id,
                request.nonce,
                request.activation.layer_id,
            )

            await self.shard.admit_frame(request)

            return ActivationResponse(
                success=True,
                message="Activation processed successfully",
                node_id=str(self.shard.node_id),
            )
        except Exception as e:
            logger.error("Error processing activation request: %s", e)
            return ActivationResponse(
                success=False,
                message=f"Error: {str(e)}",
                node_id=str(self.shard.node_id),
            )

    async def HealthCheck(
        self,
        request: HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> HealthResponse:
        """Handle health check requests"""
        logger.debug(
            "Node %s received health request from %s",
            self.shard.node_id,
            request.requester_id,
        )

        return HealthResponse(
            healthy=self.shard.adapter.running,
            node_id=str(self.shard.node_id),
            assigned_layers=self.shard.runtime.assigned_layers,
            queue_size=self.shard.runtime.activation_recv_queue.qsize(),
            active_requests=0,
        )

    async def ResetCache(
        self,
        request: ResetCacheRequest,
        context: grpc.aio.ServicerContext,
    ) -> ResetCacheResponse:
        """Handle reset cache requests"""

        try:
            logger.debug("Node %s received reset cache request", self.shard.node_id)

            # Reset the cache
            await self.shard.reset_cache()

            return ResetCacheResponse(
                success=True,
                message="Activation processed successfully",
            )
        except Exception as e:
            logger.error("Error processing reset-cache request: %s", e)
            return ResetCacheResponse(
                success=False,
                message=f"Error: {str(e)}",
            )

    async def MeasureLatency(
        self,
        request: LatencyMeasureRequest,
        context: grpc.aio.ServicerContext,
    ) -> LatencyMeasureResponse:
        """Handle latency measurement requests"""
        try:
            logger.debug(
                "Node %s received latency measurement request from %s, payload size: %s",
                self.shard.node_id,
                request.requester_id,
                request.payload_size,
            )

            return LatencyMeasureResponse(
                success=True,
                message="Latency measurement response",
                node_id=str(self.shard.node_id),
                timestamp=int(time.time() * 1000),  # Current timestamp in ms
            )
        except Exception as e:
            logger.error("Error processing latency measurement request: %s", e)
            return LatencyMeasureResponse(
                success=False,
                message=f"Error: {str(e)}",
                node_id=str(self.shard.node_id),
                timestamp=int(time.time() * 1000),
            )

    async def StreamActivations(self, request_iterator, context):
        try:
            async for frame in request_iterator:
                if frame.end_of_request:
                    # Acknowledge end-of-request with the last seen nonce (if any)
                    try:
                        yield pb2.StreamAck(
                            nonce=frame.request.nonce,
                            seq=frame.seq,
                            accepted=True,
                            message="eor",
                        )
                    except Exception:
                        pass
                    # You can choose break (close stream) or continue (keep stream alive)
                    break

                req = frame.request  # pb2.ActivationRequest
                if not req.nonce:
                    yield pb2.StreamAck(
                        nonce="", seq=frame.seq, accepted=False, message="missing nonce"
                    )
                    continue

                await self.shard.admit_frame(req)
                yield pb2.StreamAck(nonce=req.nonce, seq=frame.seq, accepted=True)

        except Exception as e:
            logger.error("[STREAM][RX] error: %s", e)
            context.abort(grpc.StatusCode.INTERNAL, str(e))
