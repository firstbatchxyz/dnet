from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List
import threading

import mlx.core as mx
from fastapi import Request
from fastapi.responses import JSONResponse
from grpc import aio as aio_grpc

from hypercorn import Config
import hypercorn.asyncio as aio_hypercorn

from ...utils.logger import logger
from ...utils.grpc_config import GRPC_AIO_OPTIONS
from ...utils.serialization import tensor_to_bytes
from ...protos import (
    dnet_ring_pb2,
    dnet_ring_pb2_grpc,
)
from .servicer import DnetRingServiceServicer
from dperf.profiler import profile_device


class StartupMixin:
    async def start(self, shutdown_trigger: Any = lambda: asyncio.Future()):
        self.running = True
        # Capture the main event loop for cross-thread scheduling
        try:
            self._loop = asyncio.get_running_loop()
        except Exception:
            self._loop = None
        await self._start_grpc_server()
        await self._start_http_server(shutdown_trigger)
        await asyncio.sleep(0.2)

        if self.next_node_address:
            await self._connect_next_node()
            if self._await_next_ready and self.next_node_channel:
                for attempt in range(1, 6):
                    try:
                        req = dnet_ring_pb2.HealthRequest(requester_id=str(self.node_id))
                        if self.next_node_stub is None:
                            raise RuntimeError("next_node_stub is None after connect")
                        resp = await asyncio.wait_for(self.next_node_stub.HealthCheck(req), timeout=3.0)
                        if resp.healthy:
                            logger.info("Next node healthy on attempt %s: %s", attempt, self.next_node_address)
                            break
                    except Exception as e:
                        logger.info("Waiting for next node health (attempt %s): %s", attempt, e)
                        await asyncio.sleep(min(1.0 * attempt, 3.0))

        self.background_tasks = [
            asyncio.create_task(self._ingress_worker()),
            asyncio.create_task(self._prefetch_worker()),
            asyncio.create_task(self._send_worker()),
        ]
        # Start idle sweeper to close silent streams
        try:
            if getattr(self, "_streaming_enabled", False) and hasattr(self, "_stream_sweeper"):
                self.background_tasks.append(asyncio.create_task(self._stream_sweeper()))
        except Exception:
            pass

        self.compute_thread = threading.Thread(target=self._compute_worker, daemon=True)
        self.compute_thread.start()

        initial_window = self._assigned_sorted[: self.window_size]
        if not (self._warmup_completed and self._warmup_keep_flag):
            for lyr in initial_window:
                self._prefetch_to_ram(lyr)
                self._enqueue_weight_prefetch(lyr)

        m = len(self._assigned_sorted)
        if m > 0:
            if m % self.window_size != 0:
                logger.warning(
                    "Window size %s does not divide local layer count %s. Rounds per token will vary; consider setting k*w = %s.",
                    self.window_size,
                    m,
                    m,
                )
            else:
                k = m // self.window_size
                logger.info("Windowed prefetch: m=%s, w=%s, k=%s rounds per token", m, self.window_size, k)

        self._start_discovery()
        logger.info("Shard node %s completed initialization on port %s", self.node_id, self.listen_port)

    def _start_discovery(self):
        from socket import gethostname
        from secrets import token_hex

        hostname = gethostname()
        instance = f"shard-{token_hex(4)}-{hostname}"
        self.discovery.create_instance(
            instance,
            hostname,
            "0.0.0.0",
            self.http_port,
            self.listen_port,
            is_manager=False,
        )
        self.discovery.start()
        logger.info("Discovery service started for shard node %s", self.node_id)

    async def _start_grpc_server(self):
        self.server = aio_grpc.server(options=GRPC_AIO_OPTIONS)
        servicer = DnetRingServiceServicer(self)
        dnet_ring_pb2_grpc.add_DnetRingServiceServicer_to_server(servicer, self.server)
        listen_addr_v6 = f"[::]:{self.listen_port}"
        listen_addr_v4 = f"0.0.0.0:{self.listen_port}"
        self.server.add_insecure_port(listen_addr_v6)
        self.server.add_insecure_port(listen_addr_v4)
        await self.server.start()
        logger.info("Shard node %s gRPC server started on %s and %s", self.node_id, listen_addr_v6, listen_addr_v4)
        try:
            await asyncio.get_running_loop().run_in_executor(self.executor, self._warmup_serialization)
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
        logger.info("[WARMUP] Starting shard warmup with window size %s", self.window_size)
        batch_size, seq_len = 1, 1
        hidden_size = self.model_metadata.model_config.get("hidden_size", 2560)
        x = mx.zeros((batch_size, seq_len, hidden_size), dtype=mx.bfloat16)
        start_time = time.perf_counter()
        try:
            default_n = max(1, int(getattr(self, "_resident_windows", 1)))
        except Exception:
            default_n = 1
        try:
            max_windows = max(1, int(os.getenv("SHARD_WARMUP_WINDOWS", str(default_n))))
        except Exception:
            max_windows = default_n
        windows: list[list[int]] = []
        for window_start in range(0, len(self._assigned_sorted), self.window_size):
            window_end = min(window_start + self.window_size, len(self._assigned_sorted))
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

    async def _start_http_server(self, shutdown_trigger: Any):
        await self._setup_routes()
        config = Config.from_mapping(
            bind=f"0.0.0.0:{self.http_port}", log_level="info", log_config=None, use_reloader=False, h2c=True
        )
        self.http_server = asyncio.create_task(aio_hypercorn.serve(self.app, config, shutdown_trigger=shutdown_trigger))  # type: ignore
        logger.info("Shard node %s HTTP server started on port %s", self.node_id, self.http_port)

    async def _setup_routes(self):
        @self.app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse(
                content={
                    "status": "ok",
                    "node_id": self.node_id,
                    "running": self.running,
                    "assigned_layers": self.assigned_layers,
                    "queue_size": self.activation_recv_queue.qsize(),
                    "grpc_port": self.listen_port,
                    "http_port": self.http_port,
                }
            )

        @self.app.post("/profile")
        async def profile(request: Request) -> JSONResponse:
            try:
                body = await request.json()
                devices = body.get("devices", {})
                payload_sizes = body.get("payload_sizes", [1024])
                latency_results = await self._measure_latency_to_devices(devices, payload_sizes)
                try:
                    device_profile = await self._profile_device()
                    device_profile_str = device_profile.json()
                except Exception as e:
                    logger.error("Failed to profile device: %s", e)
                    device_profile_str = {"error": f"Device profiling failed: {e}"}
                return JSONResponse(content={"profile": device_profile_str, "latency": latency_results})
            except Exception as e:
                logger.error("Error in /profile endpoint: %s", e)
                return JSONResponse(status_code=500, content={"error": str(e)})

        @self.app.post("/warm")
        async def warm(request: Request) -> JSONResponse:
            try:
                body = await request.json()
                start = int(body.get("start", -1))
                window = int(body.get("window", self.window_size))
                if start < 0:
                    return JSONResponse(status_code=400, content={"error": "missing/invalid start"})
                start_idx = 0
                for i, lyr in enumerate(self._assigned_sorted):
                    if lyr >= start:
                        start_idx = i
                        break
                else:
                    return JSONResponse(content={"prefetched": []})
                window_layers = self._assigned_sorted[start_idx : start_idx + max(1, window)]
                for wl in window_layers:
                    self._prefetch_to_ram(wl)
                    self._enqueue_weight_prefetch(wl)
                return JSONResponse(content={"prefetched": window_layers})
            except Exception as e:
                logger.error("/warm failed: %s", e)
                return JSONResponse(status_code=500, content={"error": str(e)})

    async def _measure_latency_to_devices(self, devices: Dict[str, Any], payload_sizes: List[int]) -> Dict[str, Any]:
        latency_results = {}
        for service_name, device_info in devices.items():
            if service_name.startswith(self.discovery.instance_name()):
                logger.debug("Skipping latency measurement to self: %s", service_name)
                continue
            if device_info.get("is_manager", False):
                logger.debug("Skipping latency measurement to manager/API: %s", service_name)
                continue
            try:
                shard_port = device_info.get("shard_port")
                host = device_info.get("host", "localhost")
                latency_results[service_name] = {"host": host, "port": shard_port}
            except Exception:
                pass
        return latency_results

    async def _profile_device(self):
        class ConfigObj:
            def __init__(self, config_dict):
                for k, v in config_dict.items():
                    setattr(self, k, v)

        config = ConfigObj(self.model_metadata.model_config)
        device_profile = profile_device(config, 0, 2)
        logger.info("Device profiling completed for node %s", self.node_id)
        return device_profile

    async def _connect_next_node(self):
        if not self.next_node_address:
            logger.info("Shard node %s is the final shard (no next node)", self.node_id)
            return True
        if self.next_node_channel:
            return True
        try:
            self.next_node_channel = aio_grpc.insecure_channel(self.next_node_address, options=GRPC_AIO_OPTIONS)
            self.next_node_stub = dnet_ring_pb2_grpc.DnetRingServiceStub(self.next_node_channel)
            if self._await_next_ready:
                try:
                    await asyncio.wait_for(self.next_node_channel.channel_ready(), timeout=10.0)
                    logger.info("Next node channel ready: %s", self.next_node_address)
                except Exception as e:
                    logger.warning("Waiting for next node channel readiness timed out: %s", e)
            return True
        except Exception as e:
            logger.warning(
                "Shard node %s failed to connect to next node %s: %s", self.node_id, self.next_node_address, e
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
            logger.info("Shard node %s successfully pinged: %s, healthy: %s", self.node_id, response.node_id, response.healthy)
            return True
        except Exception as e:
            logger.warning("Shard node %s failed to ping next node %s: %s", self.node_id, self.next_node_address, e)
            return False
