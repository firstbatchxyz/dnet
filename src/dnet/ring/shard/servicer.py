import grpc

from ...protos.dnet_ring_pb2 import (
    ActivationRequest,
    ActivationResponse,
    HealthRequest,
    HealthResponse,
    LatencyMeasureRequest,
    LatencyMeasureResponse,
    ResetCacheRequest,
    ResetCacheResponse,
)
from ...protos.dnet_ring_pb2_grpc import DnetRingServiceServicer
from ...protos import dnet_ring_pb2 as pb2
from ...utils.logger import logger


class ShardServicer(DnetRingServiceServicer):
    """gRPC servicer implementation"""

    def __init__(self, node):
        self.node = node

    async def SendActivation(
        self,
        request: ActivationRequest,
        context: grpc.aio.ServicerContext,
    ) -> ActivationResponse:
        """Handle incoming activation requests"""
        try:
            logger.debug(
                "Node %s received activation: nonce=%s, layer=%s",
                self.node.node_id,
                request.nonce,
                request.activation.layer_id,
            )

            await self.node.admit_frame(request)

            return ActivationResponse(
                success=True,
                message="Activation processed successfully",
                node_id=str(self.node.node_id),
            )
        except Exception as e:
            logger.error("Error processing activation request: %s", e)
            return ActivationResponse(
                success=False,
                message=f"Error: {str(e)}",
                node_id=str(self.node.node_id),
            )

    async def HealthCheck(
        self,
        request: HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> HealthResponse:
        """Handle health check requests"""
        logger.debug(
            "Node %s received health request from %s",
            self.node.node_id,
            request.requester_id,
        )

        return HealthResponse(
            healthy=self.node.running,
            node_id=str(self.node.node_id),
            assigned_layers=self.node.assigned_layers,
            queue_size=self.node.activation_recv_queue.qsize(),
            active_requests=0,
        )

    async def ResetCache(
        self,
        request: ResetCacheRequest,
        context: grpc.aio.ServicerContext,
    ) -> ResetCacheResponse:
        """Handle reset cache requests"""

        try:
            logger.debug("Node %s received reset cache request", self.node.node_id)

            # Reset the cache
            await self.node.reset_cache()

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
                self.node.node_id,
                request.requester_id,
                request.payload_size,
            )

            # Simply respond with success - the latency is measured by the requester
            import time

            return LatencyMeasureResponse(
                success=True,
                message="Latency measurement response",
                node_id=str(self.node.node_id),
                timestamp=int(time.time() * 1000),  # Current timestamp in ms
            )
        except Exception as e:
            logger.error("Error processing latency measurement request: %s", e)
            return LatencyMeasureResponse(
                success=False,
                message=f"Error: {str(e)}",
                node_id=str(self.node.node_id),
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

                # Expect a full ActivationRequest in the frame
                req = frame.request  # pb2.ActivationRequest
                if not req.nonce:
                    # If the client sent a frame without a nonce, reject
                    yield pb2.StreamAck(
                        nonce="", seq=frame.seq, accepted=False, message="missing nonce"
                    )
                    continue

                # Lossless server-side flow control: admit to ingress, then ACK
                await self.node.admit_frame(req)
                yield pb2.StreamAck(nonce=req.nonce, seq=frame.seq, accepted=True)

        except Exception as e:
            logger.error("[STREAM][RX] error: %s", e)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
