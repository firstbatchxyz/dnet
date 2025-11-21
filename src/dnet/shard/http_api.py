from typing import Optional, Mapping, Any
from hypercorn import Config
import hypercorn.asyncio as aio_hypercorn
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dnet.utils.logger import logger
from .shard import Shard
from dnet.utils.latency import DeviceLatencyResult, LatencyMeasurement, LatencyResults
from dnet_p2p import (
    DnetDeviceProperties,
    ThunderboltConnection,
)
import time
from .models import (
    HealthResponse,
    MeasureLatencyRequest,
    MeasureLatencyResponse,
    ShardLoadModelRequest,
    ShardLoadModelResponse,
    ShardProfileRequest,
    ShardProfileResponse,
    ShardUnloadModelResponse,
)
from dnet_p2p import AsyncDnetP2P
from dnet.protos import dnet_ring_pb2
from dnet.protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from grpc import aio as aio_grpc
from dnet.utils.profile_subproc import profile_device_via_subprocess
from distilp.common import DeviceProfile
from dnet.utils.repack import delete_repacked_layers


class HTTPServer:
    """
    HTTP API server for shard node.
    """

    def __init__(
        self, http_port: int, grpc_port: int, shard: Shard, discovery: AsyncDnetP2P
    ) -> None:
        self.shard = shard
        self.http_port: int = http_port
        self.grpc_port: int = grpc_port
        self.app = FastAPI()
        self.http_server: Optional[asyncio.Task] = None
        self.discovery = discovery

    async def start(self, shutdown_trigger: Any = lambda: asyncio.Future()) -> None:
        await self._setup_routes()

        # Start HTTP server in background
        config = Config.from_mapping(
            bind=f"0.0.0.0:{self.http_port}",
            log_level="info",
            log_config=None,
            use_reloader=False,
            h2c=False,
        )

        self.http_server = asyncio.create_task(
            aio_hypercorn.serve(self.app, config, shutdown_trigger=shutdown_trigger)  # type: ignore
        )

    async def shutdown(self) -> None:
        """Shutdown HTTP server."""
        if self.http_server and not self.http_server.done():
            self.http_server.cancel()
            try:
                await self.http_server
            except asyncio.CancelledError:
                pass
        logger.info("HTTP server on port %d stopped", self.http_port)

    async def wait_closed(self, timeout: float = 5.0) -> bool:
        if not self.http_server:
            return True
        try:
            await asyncio.wait_for(self.http_server, timeout)
            logger.info("HTTP server on port %d stopped", self.http_port)
            return True
        except asyncio.TimeoutError:
            return False

    async def _measure_latency_to_devices(
        self,
        devices: Mapping[str, DnetDeviceProperties],
        thunderbolts: Mapping[str, ThunderboltConnection],
        payload_sizes: list[int],
    ) -> LatencyResults:
        """Measure latency to all devices except self.

        Args:
            devices: Device information mapping
            thunderbolts: Thunderbolt connection information
            payload_sizes: List of payload sizes to test

        Returns:
            Latency measurement results
        """
        latency_results_dict: dict[str, DeviceLatencyResult] = {}

        for instance, device_info in devices.items():
            # Skip measuring latency to ourselves
            if instance == self.discovery.instance_name():
                logger.debug("Skipping latency measurement to self: %s", instance)
                continue

            # Skip measuring latency to API (manager) devices
            if device_info.is_manager:
                logger.debug(
                    "Skipping latency measurement to manager/API: %s", instance
                )
                continue

            try:
                shard_port = device_info.shard_port

                # Check for Thunderbolt connection
                if instance in thunderbolts:
                    tb_data = thunderbolts[instance]
                    instance_tb_ip = tb_data.ip_addr
                    logger.info(
                        "Using Thunderbolt for %s at %s, connected to instance %s",
                        instance,
                        instance_tb_ip,
                        tb_data.instance,
                    )
                else:
                    # No Thunderbolt, use WiFi
                    instance_tb_ip = device_info.local_ip

                if not shard_port or not instance_tb_ip:
                    logger.warning("No shard_port or local_ip for device %s", instance)
                    continue

                # Connect to target shard's gRPC server
                target_address = f"{instance_tb_ip}:{shard_port}"
                channel = aio_grpc.insecure_channel(target_address)

                stub = DnetRingServiceStub(channel)

                # Measure latency for each payload size
                latency_measurements: list[LatencyMeasurement] = []
                for payload_size in payload_sizes:
                    # Create dummy payload
                    dummy_data = b"x" * payload_size

                    start_time = time.perf_counter()
                    timestamp_ms = int(time.time() * 1000)

                    request = dnet_ring_pb2.LatencyMeasureRequest(
                        requester_id=str(self.shard.node_id),
                        payload_size=payload_size,
                        dummy_data=dummy_data,
                        timestamp=timestamp_ms,
                    )

                    response = await stub.MeasureLatency(request)
                    end_time = time.perf_counter()

                    if response.success:
                        latency_ms = (end_time - start_time) * 1000
                        latency_measurements.append(
                            LatencyMeasurement(
                                payload_size=payload_size,
                                latency_ms=round(latency_ms, 2),
                                success=True,
                                error=None,
                            )
                        )
                    else:
                        latency_measurements.append(
                            LatencyMeasurement(
                                payload_size=payload_size,
                                success=False,
                                error=response.message,
                                latency_ms=0,
                            )
                        )

                # Store results
                result = DeviceLatencyResult(
                    target_node_id=response.node_id if response.success else None,
                    measurements=latency_measurements,
                    success=True,
                    error=None,
                )
                latency_results_dict[instance] = result

                # Close channel
                await channel.close()

            except Exception as e:
                logger.error("Error measuring latency to %s: %s", instance, e)
                result = DeviceLatencyResult(
                    target_node_id=None,
                    success=False,
                    error=str(e),
                    measurements=[],
                )
                latency_results_dict[instance] = result

        return LatencyResults(results=latency_results_dict)

    async def _profile_device(self, repo_id: str, max_batch_exp: int) -> DeviceProfile:
        """Profile device using a profiler in a subprocess and return the object.

        Args:
            repo_id: Hugging Face repository ID
            max_batch_exp: Maximum batch size exponent (2^max_batch_exp)

        Returns:
            Device profile information as a plain dict
        """
        profile_dict = profile_device_via_subprocess(
            repo_id, max_batch_exp=max_batch_exp, debug=0
        )
        logger.info("Device profiling completed for node %s", self.shard.node_id)
        return profile_dict

    async def _setup_routes(self) -> None:
        """Setup HTTP routes."""

        @self.app.get("/health")
        async def health() -> HealthResponse:
            try:
                instance = self.discovery.instance_name()
            except Exception:
                instance = None
            return HealthResponse(
                status="ok",
                node_id=self.shard.node_id,
                running=self.shard.adapter.running,
                model_loaded=self.shard.runtime.model is not None,
                model_path=self.shard.runtime.model_path,
                assigned_layers=self.shard.runtime.assigned_layers,
                queue_size=self.shard.runtime.queue_size(),
                grpc_port=self.grpc_port,
                http_port=self.http_port,
                instance=instance,
            )

        @self.app.post("/profile")
        async def profile(req: ShardProfileRequest) -> ShardProfileResponse:
            logger.info("Received /profile request")
            try:
                device_profile = await self._profile_device(
                    req.repo_id, req.max_batch_exp
                )

                return ShardProfileResponse(profile=device_profile)
            except Exception as e:
                logger.error(f"Error in /profile endpoint: {e}")
                raise

        @self.app.post("/measure_latency")
        async def measure_latency(
            req: MeasureLatencyRequest,
        ) -> MeasureLatencyResponse:
            logger.info("Received /measure_latency request")
            try:
                # Measure latencies to other devices
                latency_results = await self._measure_latency_to_devices(
                    req.devices, req.thunderbolts, req.payload_sizes
                )

                return MeasureLatencyResponse(latency=latency_results)
            except Exception as e:
                logger.error(f"Error in /measure_latency endpoint: {e}")
                raise

        @self.app.post("/load_model")
        async def load_model_endpoint(
            req: ShardLoadModelRequest,
        ) -> ShardLoadModelResponse:
            """Load model with specified layers."""
            try:
                logger.info(
                    f"HTTP /load_model: model={req.model_path}, layers={req.layers}, "
                    f"next_node={req.next_node or 'none'}, window_size={req.window_size}, "
                    f"total_layers={req.total_layers}, kv_bits={req.kv_bits or 'default'}, "
                    f"api_callback={req.api_callback_address or 'none'}"
                )
                result = await self.shard.load_model(req)
                return result

            except Exception as e:
                logger.error(f"Error in /load_model endpoint: {e}")
                return ShardLoadModelResponse(
                    success=False,
                    message=f"Error: {str(e)}",
                    layers_loaded=[],
                    load_time_ms=0.0,
                )

        @self.app.post("/unload_model")
        async def unload_model_endpoint() -> ShardUnloadModelResponse:
            """Unload current model."""
            try:
                logger.info("HTTP /unload_model")
                result = await self.shard.unload_model()
                return result

            except Exception as e:
                logger.error(f"Error in /unload_model endpoint: {e}")
                return ShardUnloadModelResponse(
                    success=False,
                    message=f"Error: {str(e)}",
                )

        @self.app.post("/cleanup_repacked")
        async def cleanup_repacked(request: Request) -> JSONResponse:
            """Delete repacked per-layer weights on this shard to free disk.

            Body JSON (all fields optional):
              - model_id: restrict cleanup to this model bucket
              - all: when true, remove the entire repack directory base
            """
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            model_id = (payload or {}).get("model_id")
            all_flag = bool((payload or {}).get("all", False))

            try:
                removed = delete_repacked_layers(
                    model_id=model_id,
                    all_flag=all_flag,
                    current_model_path=self.shard.runtime.model_path,
                )
                return JSONResponse(content={"removed": list(removed)})
            except Exception as e:
                logger.error("/cleanup_repacked failed: %s", e)
                return JSONResponse(status_code=500, content={"error": str(e)})
