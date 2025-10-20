from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Mapping
import threading
from socket import gethostname
from secrets import token_hex

import mlx.core as mx
from fastapi import Request
from fastapi.responses import JSONResponse
from grpc import aio as aio_grpc

from hypercorn import Config
import hypercorn.asyncio as aio_hypercorn
from dnet_p2p.thunderbolt import ThunderboltConnection
from dnet_p2p import (
    DnetDeviceProperties,
    discover_thunderbolt_connection,
)

from ...protos.dnet_ring_pb2_grpc import add_DnetRingServiceServicer_to_server
from .servicer import ShardServicer
from ...utils.logger import logger
from ...utils.serialization import tensor_to_bytes
from ...utils.latency import (
    DeviceLatencyResult,
    LatencyMeasurement,
    LatencyResults,
    calculate_median_latency_seconds,
)
from .models import (
    HealthResponse,
    ShardLoadModelRequest,
    ShardLoadModelResponse,
    ShardProfileRequest,
    ShardProfileResponse,
    ShardUnloadModelResponse,
)
from ...protos import dnet_ring_pb2


class StartupMixin:
    async def start(self, shutdown_trigger: Any = lambda: asyncio.Future()):
        self.running = True
        try: # Capture the main event loop for cross-thread scheduling
            self._loop = asyncio.get_running_loop()
        except Exception:
            self._loop = None
        await self._start_grpc_server()
        await self._start_http_server(shutdown_trigger)
        await asyncio.sleep(0.2)

        self.background_tasks = [
            asyncio.create_task(self._ingress_worker()),
            asyncio.create_task(self._prefetch_worker()),
            asyncio.create_task(self._send_worker()),
        ]
        # Start idle sweeper to close silent streams
        try:
            if getattr(self, "_streaming_enabled", False) and hasattr(
                self, "_stream_sweeper"
            ):
                self.background_tasks.append(
                    asyncio.create_task(self._stream_sweeper())
                )
        except Exception:
            pass

        self.compute_thread = threading.Thread(target=self._compute_worker, daemon=True)
        self.compute_thread.start()

        self._start_discovery()
        logger.info(
            "Shard node %s started on gRPC port %s HTTP port %s",
            self.node_id,
            self.grpc_port,
            self.http_port,
        )

    def _start_discovery(self) -> None:
        """Start mDNS discovery service."""
        hostname = gethostname()
        # TODO: optionally take shard name from CLI
        instance = f"shard-{token_hex(4)}-{hostname}"
        self.discovery.create_instance(
            instance,
            hostname,
            "0.0.0.0",  # Binds to all addresses
            self.http_port,  # HTTP port
            self.grpc_port,  # gRPC port
            is_manager=False,  # Shard is never a manager
        )
        self.discovery.start()
        logger.info(
            "Discovery service started for shard node %s with name %s",
            self.node_id,
            self.discovery.fullname(),
        )

    async def _start_grpc_server(self) -> None:
        """Start gRPC server."""
        self.server = aio_grpc.server()

        # Add the ring servicer; shard acts as client for ShardApiService (to API)
        servicer = ShardServicer(self)  # type: ignore # FIXME: !!!
        add_DnetRingServiceServicer_to_server(servicer, self.server)

        listen_addr = f"[::]:{self.grpc_port}"
        self.server.add_insecure_port(listen_addr)
        await self.server.start()
        logger.info(
            "Shard node %s gRPC server started on %s", self.node_id, listen_addr
        )
        try:
            await asyncio.get_running_loop().run_in_executor(
                self.executor, self._warmup_serialization
            )
            logger.info("Warmup serialization completed")
        except Exception as e:
            logger.warning("Warmup serialization failed: %s", e)

    def _warmup_serialization(self):
        try:
            dummy = mx.random.normal((1024, 1024), dtype=mx.float32)
            dummy16 = dummy.astype(self._wire_mx_dtype)
            _ = tensor_to_bytes(dummy16)
        except Exception:
            pass

    def _warmup_shard(self):
        logger.info(
            "[WARMUP] Starting shard warmup with window size %s", self.window_size
        )
        batch_size, seq_len = 1, 1
        hidden_size = self.model_metadata.model_config.get("hidden_size", 2560)
        x = mx.zeros((batch_size, seq_len, hidden_size), dtype=mx.bfloat16)
        start_time = time.perf_counter()
        try:
            default_n = max(1, int(getattr(self, "_resident_windows", 1)))
        except Exception:
            default_n = 1
        try:
            max_windows = max(
                1,
                int(
                    getattr(self, "config", None).warmup_windows
                    if getattr(self, "config", None)
                    else default_n
                ),
            )
        except Exception:
            max_windows = default_n
        windows: list[list[int]] = []
        for window_start in range(0, len(self._assigned_sorted), self.window_size):
            window_end = min(
                window_start + self.window_size, len(self._assigned_sorted)
            )
            windows.append(self._assigned_sorted[window_start:window_end])
        for wi, window_layers in enumerate(windows[:max_windows]):
            weights_to_bind = {}
            for layer_id in window_layers:
                weights = self.weight_cache.get_weight(layer_id)
                if weights:
                    for k, v in weights.items():
                        weights_to_bind[k] = v
            if weights_to_bind:
                self.model.load_weights(list(weights_to_bind.items()), strict=False)
            try:
                for layer_id in window_layers:
                    x = self.model.apply_single_layer(layer_id, x, cache=None)
                    _s = mx.sum(x)
                    mx.eval(_s)
            except Exception:
                pass
            try:
                for lid in window_layers:
                    self.weight_cache.decrease_reference(lid)
            except Exception:
                pass
            if not self._warmup_keep_flag:
                try:
                    if hasattr(self.model, "unload_layers"):
                        self.model.unload_layers(window_layers)  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    self.weight_cache.evict_layers(window_layers)
                except Exception:
                    pass
        total_time = (time.perf_counter() - start_time) * 1000
        self._warmup_completed = True
        logger.info(
            "[WARMUP] Shard warmup completed in %.2fms; windows=%s kept=%s",
            total_time,
            min(len(windows), max_windows),
            int(self._warmup_keep_flag),
        )

    async def _start_http_server(self, shutdown_trigger: Any) -> None:
        """Start HTTP server.

        Args:
            shutdown_trigger: Shutdown trigger function
        """
        await self._setup_routes()

        # Start HTTP server in background
        config = Config.from_mapping(
            bind=f"0.0.0.0:{self.http_port}",
            log_level="info",
            log_config=None,
            use_reloader=False,
            h2c=False,
        )

        # Start the server as a background task
        self.http_server = asyncio.create_task(
            aio_hypercorn.serve(self.app, config, shutdown_trigger=shutdown_trigger)  # type: ignore
        )
        logger.info(
            "Shard node %s HTTP server started on port %s", self.node_id, self.http_port
        )

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
                node_id=self.node_id,
                running=self.running,
                model_loaded=self._check_model_loaded(),
                model_path=self.model_path,
                assigned_layers=self.assigned_layers,
                queue_size=self.activation_recv_queue.qsize(),
                grpc_port=self.grpc_port,
                http_port=self.http_port,
                instance=instance,
            )

        @self.app.post("/profile")
        async def profile(req: ShardProfileRequest) -> ShardProfileResponse:
            logger.info("Received /profile request")
            try:
                # Measure latencies
                latency_results = await self._measure_latency_to_devices(
                    req.devices, req.thunderbolts, req.payload_sizes
                )

                # Profile device using dperf
                device_profile = await self._profile_device(
                    req.repo_id, req.max_batch_exp
                )

                # Overwrite `t_comm` with median latency (subprocess returns a dict)
                median_latency = calculate_median_latency_seconds(latency_results)
                if median_latency is not None:
                    device_profile["t_comm"] = float(median_latency)
                    logger.info(
                        f"Set t_comm to median latency: {device_profile['t_comm']:.6f}s"
                    )
                else:
                    logger.warning(
                        "No valid latency measurements, keeping default t_comm"
                    )

                # Return the dict payload directly
                return ShardProfileResponse(
                    profile=device_profile,
                    latency=latency_results,
                )
            except Exception as e:
                logger.error(f"Error in /profile endpoint: {e}")
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
                    f"total_layers={req.total_layers}, api_callback={req.api_callback_address or 'none'}"
                )
                result = await self.load_model(req)
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
                result = await self.unload_model()
                return result

            except Exception as e:
                logger.error(f"Error in /unload_model endpoint: {e}")
                return ShardUnloadModelResponse(
                    success=False,
                    message=f"Error: {str(e)}",
                )

        @self.app.post("/warm")
        async def warm(request: Request) -> JSONResponse:
            try:
                body = await request.json()
                start = int(body.get("start", -1))
                window = int(body.get("window", self.window_size))
                if start < 0:
                    return JSONResponse(
                        status_code=400, content={"error": "missing/invalid start"}
                    )
                start_idx = 0
                for i, lyr in enumerate(self._assigned_sorted):
                    if lyr >= start:
                        start_idx = i
                        break
                else:
                    return JSONResponse(content={"prefetched": []})
                window_layers = self._assigned_sorted[
                    start_idx : start_idx + max(1, window)
                ]
                for wl in window_layers:
                    self._prefetch_to_ram(wl)
                    self._enqueue_weight_prefetch(wl)
                return JSONResponse(content={"prefetched": window_layers})
            except Exception as e:
                logger.error("/warm failed: %s", e)
                return JSONResponse(status_code=500, content={"error": str(e)})

    async def _profile_device(self, repo_id: str, max_batch_exp: int) -> dict:
        """Profile device using dperf in a subprocess and return a dict.

        Args:
            repo_id: Hugging Face repository ID
            max_batch_exp: Maximum batch size exponent (2^max_batch_exp)

        Returns:
            Device profile information as a plain dict
        """
        from ...utils.profile_subproc import profile_device_via_subprocess

        profile_dict = profile_device_via_subprocess(
            repo_id, max_batch_exp=max_batch_exp, debug=0
        )
        logger.info("Device profiling completed for node %s", self.node_id)
        return profile_dict

    async def _connect_next_node(self) -> bool:
        """Connect to next node in ring.

        Returns:
            True if connected or no next node, False on failure
        """
        if not self.next_node:
            logger.info(f"Shard node {self.node_id} is the final shard (no next node)")
            return True

        if self.next_node_channel:
            logger.debug(f"Shard node {self.node_id} already connected to next node.")
            return True

        try:
            # use thunderbolt here if available
            this_properties = self.discovery.get_own_properties()
            thunderbolt_conn = discover_thunderbolt_connection(
                this_properties,
                self.next_node,
            )
            next_ip = (
                thunderbolt_conn.ip_addr
                if thunderbolt_conn
                else self.next_node.local_ip
            )
            address = f"{next_ip}:{self.next_node.shard_port}"
            logger.info(
                f"Shard node {this_properties.instance} connecting to next node {self.next_node.instance} at {address}"
            )

            self.next_node_channel = aio_grpc.insecure_channel(address)
            from ...protos.dnet_ring_pb2_grpc import DnetRingServiceStub

            self.next_node_stub = DnetRingServiceStub(self.next_node_channel)
            return True
        except Exception as e:
            logger.warning(
                f"Shard node {self.node_id} failed to connect to next node {address}: {e}"
            )
            self.next_node_channel = None
            self.next_node_stub = None
            return False

    async def _reconnect_next_node(self) -> bool:
        try:
            if self.next_node_channel:
                await self.next_node_channel.close()
        except Exception:
            pass
        self.next_node_channel = None
        self.next_node_stub = None
        return await self._connect_next_node()

    async def _health_check(self):
        try:
            health_request = dnet_ring_pb2.HealthRequest(requester_id=str(self.node_id))
            response = await self.next_node_stub.HealthCheck(health_request)  # type: ignore
            logger.info(
                "Shard node %s successfully pinged: %s, healthy: %s",
                self.node_id,
                response.node_id,
                response.healthy,
            )
            return True
        except Exception as e:
            logger.warning(
                "Shard node %s failed to ping next node %s: %s",
                self.node_id,
                self.next_node_address,
                e,
            )
            return False

    async def _measure_latency_to_devices(
        self,
        devices: Mapping[str, DnetDeviceProperties],
        thunderbolts: Mapping[str, ThunderboltConnection],
        payload_sizes: List[int],
    ) -> LatencyResults:
        """Measure latency to all devices except self.

        Args:
            devices: Device information mapping
            thunderbolts: Thunderbolt connection information
            payload_sizes: List of payload sizes to test

        Returns:
            Latency measurement results
        """
        latency_results_dict: Dict[str, DeviceLatencyResult] = {}

        for service_name, device_info in devices.items():
            # Skip measuring latency to ourselves
            if service_name.startswith(self.discovery.instance_name()):
                logger.debug("Skipping latency measurement to self: %s", service_name)
                continue

            # Skip measuring latency to API (manager) devices
            if device_info.is_manager:
                logger.debug(
                    "Skipping latency measurement to manager/API: %s", service_name
                )
                continue

            try:
                shard_port = device_info.shard_port

                # Check for Thunderbolt connection
                if service_name in thunderbolts:
                    tb_data = thunderbolts[service_name]
                    service_ip = tb_data.ip_addr
                    logger.info(
                        "Using Thunderbolt for %s at %s, connected to instance %s",
                        service_name,
                        service_ip,
                        tb_data.instance,
                    )
                else:
                    # No Thunderbolt, use WiFi
                    service_ip = device_info.local_ip

                if not shard_port or not service_ip:
                    logger.warning(
                        "No shard_port or local_ip for device %s", service_name
                    )
                    continue

                # Connect to target shard's gRPC server
                target_address = f"{service_ip}:{shard_port}"
                channel = aio_grpc.insecure_channel(target_address)
                from ...protos.dnet_ring_pb2_grpc import DnetRingServiceStub

                stub = DnetRingServiceStub(channel)

                # Measure latency for each payload size
                latency_measurements: List[LatencyMeasurement] = []
                for payload_size in payload_sizes:
                    # Create dummy payload
                    dummy_data = b"x" * payload_size

                    start_time = time.perf_counter()
                    timestamp_ms = int(time.time() * 1000)

                    request = dnet_ring_pb2.LatencyMeasureRequest(
                        requester_id=str(self.node_id),
                        payload_size=payload_size,
                        dummy_data=dummy_data,
                        timestamp=timestamp_ms,
                    )

                    response = await stub.MeasureLatency(request)  # type: ignore
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
                latency_results_dict[service_name] = result

                # Close channel
                await channel.close()

            except Exception as e:
                logger.error("Error measuring latency to %s: %s", service_name, e)
                result = DeviceLatencyResult(
                    target_node_id=None,
                    success=False,
                    error=str(e),
                    measurements=[],
                )
                latency_results_dict[service_name] = result

        return LatencyResults(results=latency_results_dict)
