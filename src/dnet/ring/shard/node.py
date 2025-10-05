"""Ring shard node implementation with dynamic model loading."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from queue import Empty, Queue
from secrets import token_hex
from socket import gethostname
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast
from urllib.parse import urlparse

import grpc
import httpx
import mlx.core as mx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from grpc import aio as aio_grpc
from hypercorn import Config
import hypercorn.asyncio as aio_hypercorn

from dnet_p2p import DnetP2P, ThunderboltInstance
from dperf import DeviceProfileInfo, profile_device

from ...protos import dnet_ring_pb2, shard_api_comm_pb2
from ...protos.dnet_ring_pb2_grpc import add_DnetRingServiceServicer_to_server
from ...protos.shard_api_comm_pb2_grpc import (
    ShardApiServiceStub,
    add_ShardApiServiceServicer_to_server,
)
from ...utils.async_utils import make_cache
from ...utils.latency import (
    DeviceLatencyResult,
    LatencyMeasurement,
    LatencyResults,
    calculate_median_latency_seconds,
)
from ...utils.logger import logger
from ...utils.model import ModelMetadata, get_model_metadata
from ...utils.serialization import dtype_map, mlx_dtype_map, tensor_to_bytes
from ..api_models import RecieveResultRequest
from ..data_types import ActivationMessage
from ..memory_pool import LayerAwareMemoryPool
from ..model import get_ring_model
from ..weight_cache import WeightCache
from .servicer import ShardServicer


def utc_epoch_now() -> int:
    """Return current UTC timestamp in milliseconds."""
    return int(time.time() * 1000)


class RingShardNode:
    """Single shard node in the distributed inference ring with dynamic model loading."""

    def __init__(
        self,
        node_id: int,
        listen_port: int,
        http_port: int,
        queue_size: int = 10,
        prefetch_window_size: int = 2,
    ) -> None:
        """Initialize ring shard node.

        Args:
            node_id: Node identifier
            listen_port: gRPC listen port
            http_port: HTTP server port
            queue_size: Size of activation processing queue
            prefetch_window_size: Number of layers to prefetch ahead
        """
        self.node_id = node_id
        self.listen_port = listen_port
        self.http_port = http_port
        self.queue_size = queue_size
        self.prefetch_window_size = max(1, int(prefetch_window_size or 1))

        # Model state (loaded dynamically)
        self.model_metadata: Optional[ModelMetadata] = None
        self.assigned_layers: List[int] = []
        self.model: Optional[Any] = None  # Ring model instance
        self.cache: Optional[Any] = None  # KV cache
        self.model_path: Optional[str] = None  # Track currently loaded model path

        # Topology (configured later)
        self.next_node_address: Optional[str] = None

        # HTTP server
        self.app = FastAPI()
        self.http_server: Optional[asyncio.Task] = None

        # Memory management (initialized when model loads)
        self.input_pool: Optional[LayerAwareMemoryPool] = None
        self.output_pool: Optional[LayerAwareMemoryPool] = None
        self.weight_cache: Optional[WeightCache] = None
        self._prefetch_scheduled: set[int] = set()

        # Queues for async processing
        self.activation_recv_queue: Queue[ActivationMessage] = Queue(maxsize=queue_size)
        self.weight_prefetch_queue: Queue[int] = Queue(maxsize=50)
        self.activation_computed_queue: Queue[ActivationMessage] = Queue(
            maxsize=queue_size
        )

        # Threading
        self.compute_thread: Optional[threading.Thread] = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._active_nonce: Optional[str] = None
        self._bound_versions: Dict[int, int] = {}

        # gRPC
        self.server: Optional[aio_grpc.Server] = None
        self.next_node_channel: Optional[aio_grpc.Channel] = None
        self.next_node_stub: Optional[Any] = None
        self.api_channel: Optional[aio_grpc.Channel] = None
        self.api_stub: Optional[ShardApiServiceStub] = None
        self.api_address: Optional[str] = None

        # Discovery
        self.discovery = DnetP2P("lib/dnet-p2p/lib")

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

        logger.info(
            f"Shard node {node_id} initialized with queue_size={queue_size}, "
            f"prefetch_window_size={prefetch_window_size}"
        )

    async def load_model(
        self, model_path: str, layers: List[int], warmup: bool = False
    ) -> Dict[str, Any]:
        """Load model with specified layers.

        Args:
            model_path: Path to model (HF repo ID or local path)
            layers: Layer indices to load
            warmup: Whether to perform warmup after loading

        Returns:
            Dict with success, message, layers_loaded, load_time_ms
        """
        try:
            start_time = time.perf_counter()

            # Check if already loaded with same configuration
            if (
                self.model is not None
                and self.model_path == model_path
                and self.assigned_layers == layers
            ):
                logger.info(
                    f"Node {self.node_id}: Model already loaded with same configuration"
                )
                return {
                    "success": True,
                    "message": "Model already loaded",
                    "layers_loaded": layers,
                    "load_time_ms": 0.0,
                }

            # If model loaded with different config, unload first
            if self.model is not None and (
                self.model_path != model_path or self.assigned_layers != layers
            ):
                logger.info(
                    f"Node {self.node_id}: Unloading current model to load new configuration"
                )
                await self.unload_model()

            # Load model metadata
            self.model_metadata = get_model_metadata(model_path)
            self.assigned_layers = layers
            self.model_path = model_path

            # Initialize memory pools
            self.input_pool = LayerAwareMemoryPool(total_memory_mb=512)
            self.output_pool = LayerAwareMemoryPool(total_memory_mb=512)

            # Initialize weight cache
            self.weight_cache = WeightCache(
                self.assigned_layers, self.model_metadata, window_size=self.prefetch_window_size
            )

            # Load the model
            self.model = get_ring_model(
                self.model_metadata.model_type,
                self.model_metadata.model_config,
                assigned_layers=self.assigned_layers,
            )
            self.model.eval()
            self.cache = make_cache(self.model)

            # Reset prefetch tracking
            self._prefetch_scheduled = set()
            self._bound_versions = {}

            # Warmup if requested
            if warmup:
                await self._warmup_model()

            load_time_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                f"Node {self.node_id}: Successfully loaded model {model_path} "
                f"with layers {layers} in {load_time_ms:.2f}ms"
            )

            return {
                "success": True,
                "message": f"Model loaded successfully",
                "layers_loaded": layers,
                "load_time_ms": load_time_ms,
            }

        except Exception as e:
            logger.exception(f"Node {self.node_id}: Error loading model: {e}")
            return {
                "success": False,
                "message": f"Error loading model: {str(e)}",
                "layers_loaded": [],
                "load_time_ms": 0.0,
            }

    async def unload_model(self) -> Dict[str, Any]:
        """Unload current model and free resources.

        Returns:
            Dict with success and message
        """
        try:
            if self.model is None:
                return {
                    "success": True,
                    "message": "No model loaded",
                }

            logger.info(f"Node {self.node_id}: Unloading model")

            # Clear model and cache
            self.model = None
            self.cache = None
            self.model_metadata = None
            self.assigned_layers = []
            self.model_path = None

            # Clear memory pools
            if self.weight_cache:
                # Clear all cached weights
                for layer_id in list(self._bound_versions.keys()):
                    try:
                        self.weight_cache.evict_layer(layer_id)
                    except Exception:
                        pass
                self.weight_cache = None

            self.input_pool = None
            self.output_pool = None
            self._prefetch_scheduled = set()
            self._bound_versions = {}

            # Run garbage collection to free memory
            import gc

            gc.collect()

            logger.info(f"Node {self.node_id}: Model unloaded successfully")

            return {
                "success": True,
                "message": "Model unloaded successfully",
            }

        except Exception as e:
            logger.exception(f"Node {self.node_id}: Error unloading model: {e}")
            return {
                "success": False,
                "message": f"Error unloading model: {str(e)}",
            }

    async def _warmup_model(self) -> None:
        """Warmup model by prefetching initial layers."""
        if not self.assigned_layers or not self.weight_cache:
            return

        logger.info(f"Node {self.node_id}: Starting model warmup")

        # Warm up initial local window: prefetch to RAM and enqueue device loads
        ordered = sorted(self.assigned_layers)
        initial_window = ordered[: self.prefetch_window_size]

        for lyr in initial_window:
            self._prefetch_to_ram(lyr)
            self._enqueue_weight_prefetch(lyr)

        logger.info(
            f"Node {self.node_id}: Warmup completed for layers {initial_window}"
        )

    def _check_model_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None and self.model_metadata is not None

    async def start(
        self, shutdown_trigger: Any = lambda: asyncio.Future()
    ) -> None:
        """Start the inference node.

        Args:
            shutdown_trigger: Shutdown trigger function
        """
        self.running = True

        # Start gRPC server
        await self._start_grpc_server()

        # Start HTTP server
        await self._start_http_server(shutdown_trigger)

        # Allow brief startup settling
        await asyncio.sleep(0.2)

        # Connect to next node if configured
        if self.next_node_address:
            await self._connect_next_node()

        # Start background coroutines
        self.background_tasks = [
            asyncio.create_task(self._prefetch_worker()),
            asyncio.create_task(self._send_worker()),
        ]

        # Start compute thread
        self.compute_thread = threading.Thread(target=self._compute_worker)
        self.compute_thread.start()

        # Start discovery service
        self._start_discovery()

        logger.info(
            f"Shard node {self.node_id} started on gRPC port {self.listen_port}, "
            f"HTTP port {self.http_port}"
        )

    def _start_discovery(self) -> None:
        """Start mDNS discovery service."""
        hostname = gethostname()
        instance = f"shard-{token_hex(4)}-{hostname}"
        self.discovery.create_instance(
            instance,
            hostname,
            "0.0.0.0",  # Binds to all addresses
            self.http_port,  # HTTP port
            self.listen_port,  # gRPC port
            is_manager=False,  # Shard is never a manager
        )
        self.discovery.start()
        logger.info(f"Discovery service started for shard node {self.node_id}")

    async def _start_grpc_server(self) -> None:
        """Start gRPC server."""
        self.server = aio_grpc.server()

        # Add the servicer (handles both ring and shard API services)
        servicer = ShardServicer(self)
        add_DnetRingServiceServicer_to_server(servicer, self.server)
        add_ShardApiServiceServicer_to_server(servicer, self.server)

        listen_addr = f"[::]:{self.listen_port}"
        self.server.add_insecure_port(listen_addr)
        await self.server.start()
        logger.info(
            f"Shard node {self.node_id} gRPC server started on {listen_addr}"
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
            h2c=True,
        )

        # Start the server as a background task
        self.http_server = asyncio.create_task(
            aio_hypercorn.serve(self.app, config, shutdown_trigger=shutdown_trigger)  # type: ignore
        )
        logger.info(
            f"Shard node {self.node_id} HTTP server started on port {self.http_port}"
        )

    async def _setup_routes(self) -> None:
        """Setup HTTP routes."""

        @self.app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse(
                content={
                    "status": "ok",
                    "node_id": self.node_id,
                    "running": self.running,
                    "model_loaded": self._check_model_loaded(),
                    "model_path": self.model_path,
                    "assigned_layers": self.assigned_layers,
                    "queue_size": self.activation_recv_queue.qsize(),
                    "grpc_port": self.listen_port,
                    "http_port": self.http_port,
                }
            )

        @self.app.post("/profile")
        async def profile(request: Request) -> JSONResponse:
            logger.info("Received /profile request")
            try:
                body = await request.json()
                devices: Dict[str, Any] = body.get("devices", {})
                payload_sizes: List[int] = body.get("payload_sizes", [1024])
                max_batch_exp: int = body.get("max_batch_exp", 2)
                repo_id: str = body.get("repo_id")
                thunderbolts: Dict[str, Any] = body.get(
                    "thunderbolts", {}
                )  # Thunderbolt connections FROM this device

                # Measure latencies
                latency_results = await self._measure_latency_to_devices(
                    devices, thunderbolts, payload_sizes
                )

                # Profile device using dperf
                try:
                    device_profile = await self._profile_device(repo_id, max_batch_exp)

                    # Overwrite t_comm with median latency
                    median_latency = calculate_median_latency_seconds(latency_results)
                    if median_latency is not None:
                        device_profile.t_comm = median_latency
                        logger.info(
                            f"Set t_comm to median latency: {device_profile.t_comm:.6f}s"
                        )
                    else:
                        logger.warning(
                            "No valid latency measurements, keeping default t_comm"
                        )
                    device_profile_dict = asdict(device_profile)

                except Exception as e:
                    logger.error(f"Failed to profile device: {e}")
                    device_profile_dict = {"error": f"Device profiling failed: {e}"}

                return JSONResponse(
                    content={
                        "profile": device_profile_dict,
                        "latency": latency_results.model_dump(),
                    }
                )
            except Exception as e:
                logger.error(f"Error in /profile endpoint: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

    async def _measure_latency_to_devices(
        self,
        devices: Mapping[str, Any],
        thunderbolts: Mapping[str, Any],
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
                logger.debug(f"Skipping latency measurement to self: {service_name}")
                continue

            # Skip measuring latency to API (manager) devices
            if device_info.get("is_manager", False):
                logger.debug(
                    f"Skipping latency measurement to manager/API: {service_name}"
                )
                continue

            try:
                shard_port = device_info.get("shard_port")

                # Check for Thunderbolt connection
                if service_name in thunderbolts:
                    tb_data = thunderbolts[service_name]
                    service_ip = tb_data["ip"]
                    tb_instance = ThunderboltInstance(**tb_data["instance"])
                    logger.info(
                        f"Using Thunderbolt for {service_name} at {service_ip}, "
                        f"connected to instance {tb_instance.device}"
                    )
                else:
                    # No Thunderbolt, use WiFi
                    service_ip = device_info.get("local_ip")

                if not shard_port or not service_ip:
                    logger.warning(
                        f"No shard_port or local_ip for device {service_name}"
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
                logger.error(f"Error measuring latency to {service_name}: {e}")
                result = DeviceLatencyResult(
                    target_node_id=None,
                    success=False,
                    error=str(e),
                    measurements=[],
                )
                latency_results_dict[service_name] = result

        return LatencyResults(results=latency_results_dict)

    async def _profile_device(
        self, repo_id: str, max_batch_exp: int
    ) -> DeviceProfileInfo:
        """Profile device using dperf.

        Args:
            repo_id: Hugging Face repository ID
            max_batch_exp: Maximum batch size exponent (2^max_batch_exp)

        Returns:
            Device profile information
        """
        device_profile: DeviceProfileInfo = profile_device(
            repo_id, max_batch_exp=max_batch_exp
        )
        logger.info(f"Device profiling completed for node {self.node_id}")
        return device_profile

    async def _connect_next_node(self) -> bool:
        """Connect to next node in ring.

        Returns:
            True if connected or no next node, False on failure
        """
        if not self.next_node_address:
            logger.info(
                f"Shard node {self.node_id} is the final shard (no next node)"
            )
            return True

        if self.next_node_channel:
            return True

        try:
            self.next_node_channel = aio_grpc.insecure_channel(
                self.next_node_address
            )
            from ...protos.dnet_ring_pb2_grpc import DnetRingServiceStub

            self.next_node_stub = DnetRingServiceStub(self.next_node_channel)
            return True
        except Exception as e:
            logger.warning(
                f"Shard node {self.node_id} failed to connect to next node "
                f"{self.next_node_address}: {e}"
            )
            self.next_node_channel = None
            self.next_node_stub = None
            return False

    async def reset_cache(self) -> None:
        """Reset LLM KV cache."""
        if not self._check_model_loaded():
            logger.warning(
                f"Node {self.node_id}: Cannot reset cache - no model loaded"
            )
            return

        try:
            self.cache = make_cache(self.model)
            logger.info(f"Node {self.node_id}: Cache reset successfully")
        except Exception as e:
            logger.error(f"Node {self.node_id}: Error resetting cache: {e}")

    async def receive_activation(
        self, request: dnet_ring_pb2.ActivationRequest
    ) -> None:
        """Receive activation from previous node.

        Args:
            request: Activation request from previous node
        """
        if not self._check_model_loaded():
            logger.error(
                f"Node {self.node_id}: Cannot process activation - no model loaded. "
                f"Nonce: {request.nonce}"
            )
            return

        if not (await self._connect_next_node()):
            logger.error(
                f"Node {self.node_id}: Not connected to next node "
                f"{self.next_node_address}. Dropping activation {request.nonce}"
            )
            return

        try:
            # Check if this activation is for our layers
            activation = request.activation
            target_layer = activation.layer_id + 1

            # Transport and payload metrics
            try:
                payload_bytes = len(activation.data)
            except Exception:
                payload_bytes = -1

            transport_ms = float(utc_epoch_now() - request.timestamp)
            logger.info(
                f"[PROFILE][RX] node={self.node_id} nonce={request.nonce} "
                f"target_layer={target_layer} transport_ms={transport_ms:.1f} "
                f"payload_kb={(payload_bytes/1024):.1f}"
            )

            # New sequence detection (reset KV cache once per nonce)
            if request.nonce != self._active_nonce:
                self._active_nonce = request.nonce
                await self.reset_cache()

            if target_layer in self.assigned_layers:
                # Schedule full window RAM prefetch
                next_locals = self._next_local_layers(
                    target_layer, self.prefetch_window_size - 1
                )
                window_layers = [target_layer] + next_locals
                for wl in window_layers:
                    self._prefetch_to_ram(wl)

                # Kick off device-level weight load for first layer
                self._enqueue_weight_prefetch(target_layer)

                logger.info(
                    f"Processing activation for layer {target_layer}, "
                    f"nonce: {request.nonce}"
                )

                # Allocate input pool buffer
                t_alloc = time.perf_counter()
                pool_id = self.input_pool.allocate_for_layer(  # type: ignore
                    layer_id=activation.layer_id,
                    dtype=mlx_dtype_map[activation.dtype],
                    shape=cast(Tuple[int, ...], tuple(activation.shape)),
                )
                if pool_id is None:
                    logger.warning(
                        f"Failed to allocate input pool buffer for nonce {request.nonce}"
                    )
                    return

                # Copy data to pool buffer
                buffer = self.input_pool.get_buffer(pool_id)  # type: ignore
                if buffer is not None:
                    # Convert bytes to numpy array and copy to buffer
                    data = request.activation.data
                    input_data = np.frombuffer(
                        data, dtype=dtype_map[activation.dtype]
                    )
                    buffer[: len(input_data)] = input_data
                    alloc_copy_ms = (time.perf_counter() - t_alloc) * 1000.0
                    logger.info(
                        f"[PROFILE][RX] node={self.node_id} nonce={request.nonce} "
                        f"alloc_copy_ms={alloc_copy_ms:.3f}"
                    )

                    # Update activation message with pool_id
                    activation_msg = ActivationMessage.from_proto(request, pool_id)

                    # Queue for processing
                    try:
                        self.activation_recv_queue.put_nowait(activation_msg)
                        logger.debug(
                            f"Queued activation for processing: nonce {activation_msg.nonce}"
                        )
                    except Exception:
                        logger.error(
                            f"Failed to queue activation {activation_msg.nonce} - queue full"
                        )
                        self.input_pool.release(pool_id)  # type: ignore
            else:
                # Forward to next node
                logger.debug(
                    f"Forwarding activation (layer {target_layer}) to next node, "
                    f"nonce: {request.nonce}"
                )
                await self._forward_activation(request)

        except Exception as e:
            logger.exception(f"Error receiving activation: {e}")

    def _prefetch_to_ram(self, layer_id: int) -> None:
        """Prefetch layer weights to RAM.

        Args:
            layer_id: Layer to prefetch
        """
        if layer_id not in self._prefetch_scheduled and self.weight_cache:
            self._prefetch_scheduled.add(layer_id)
            self.weight_cache.prefetch_to_ram(layer_id)

    def _enqueue_weight_prefetch(self, layer_id: int) -> None:
        """Enqueue layer for GPU weight prefetch.

        Args:
            layer_id: Layer to enqueue
        """
        try:
            self.weight_prefetch_queue.put(layer_id, timeout=0.01)
        except Exception:
            # Queue may be full; it's fine to skip
            pass

    def _next_local_layers(self, after_layer: int, count: int) -> List[int]:
        """Get next local layers after specified layer.

        Args:
            after_layer: Layer to start from
            count: Number of layers to return

        Returns:
            List of next local layer IDs
        """
        if count <= 0:
            return []
        ordered = sorted(self.assigned_layers)
        nxt = [lyr for lyr in ordered if lyr > after_layer]
        return nxt[:count]

    async def _forward_activation(
        self, request: dnet_ring_pb2.ActivationRequest
    ) -> None:
        """Forward activation to next node.

        Args:
            request: Activation request to forward
        """
        try:
            t0 = time.perf_counter()
            response = await self.next_node_stub.SendActivation(request)  # type: ignore
            rpc_ms = (time.perf_counter() - t0) * 1000.0
            if not response.success:
                logger.warning(f"Next node reported error: {response.message}")
            logger.info(
                f"[PROFILE][TX] node={self.node_id} nonce={request.nonce} "
                f"forwarded_layer={request.activation.layer_id + 1} rpc_ms={rpc_ms:.2f}"
            )
        except Exception as e:
            logger.error(f"Failed to forward activation: {e}")

    async def _prefetch_worker(self) -> None:
        """Async worker for weight prefetching."""
        while self.running:
            try:
                # Non-blocking fetch
                layer_id = self.weight_prefetch_queue.get_nowait()

                if self.weight_cache:
                    # Prefetch in background
                    await asyncio.get_running_loop().run_in_executor(
                        self.executor, self.weight_cache.get_weight, layer_id
                    )
                    logger.debug(f"Prefetched weights for layer {layer_id}")

            except Empty:
                await asyncio.sleep(0.005)
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
                await asyncio.sleep(0.02)

    def _compute_worker(self) -> None:
        """Compute thread worker."""
        while self.running:
            try:
                # Get activation from queue (blocks until available)
                activation_msg = self.activation_recv_queue.get(timeout=1.0)

                # Process the activation
                self._process_activation(activation_msg)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Compute worker error: {e}")

    def _process_activation(self, activation_msg: ActivationMessage) -> None:
        """Process activation through consecutive local layers.

        Args:
            activation_msg: Activation message to process
        """
        if not self._check_model_loaded() or not self.weight_cache or not self.input_pool or not self.output_pool:
            logger.error(
                f"Node {self.node_id}: Cannot process activation - model not loaded"
            )
            return

        try:
            # Get input activation from pool
            input_buffer = self.input_pool.get_buffer(activation_msg.pool_id)
            if input_buffer is None:
                logger.error(f"Failed to get input buffer {activation_msg.pool_id}")
                return

            # Prepare input activation as MLX array view
            input_size = int(np.prod(activation_msg.shape))
            x = (input_buffer.flatten()[:input_size]).reshape(activation_msg.shape)

            # Compute up to prefetch_window_size consecutive local layers
            start_time = time.perf_counter()
            processed = 0
            current_layer = activation_msg.layer_id + 1
            last_layer = current_layer - 1

            # Determine contiguous local window
            window_layers: List[int] = []
            _tmp_layer = current_layer
            while processed < self.prefetch_window_size and (
                _tmp_layer in self.assigned_layers
            ):
                window_layers.append(_tmp_layer)
                _tmp_layer += 1
                processed += 1

            # Ensure weights for window are resident and bind only if arrays changed
            to_bind: Dict[str, mx.array] = {}
            for wl in window_layers:
                weights = self.weight_cache.get_weight(wl)
                if weights is None:
                    logger.error(f"Failed to load weights for layer {wl}")
                    self.input_pool.release(activation_msg.pool_id)
                    return
                try:
                    # Use identity of first array as version fingerprint
                    first_arr = next(iter(weights.values()))
                    version = id(first_arr)
                except StopIteration:
                    version = -1
                if self._bound_versions.get(wl) != version:
                    for k, v in weights.items():
                        to_bind[k] = v
                    self._bound_versions[wl] = version

            if to_bind:
                self.model.load_weights(list(to_bind.items()), strict=False)

            # Run window compute
            processed = 0
            current_layer = activation_msg.layer_id + 1
            while processed < len(window_layers):
                logger.info(
                    f"Computing layer {current_layer} for nonce {activation_msg.nonce}"
                )
                t_l = time.perf_counter()
                x = self.model.apply_single_layer(current_layer, x, cache=self.cache)
                t_l_done = time.perf_counter()
                logger.info(
                    f"[PROFILE][LAYER] node={self.node_id} nonce={activation_msg.nonce} "
                    f"layer={current_layer} compute_ms={(t_l_done - t_l) * 1000.0:.3f}"
                )

                # Advance
                self.weight_cache.decrease_reference(current_layer)
                last_layer = current_layer
                current_layer += 1
                processed += 1

            computation_time = time.perf_counter() - start_time
            logger.info(
                f"Completed layers up to {last_layer} in {computation_time:.3f}s, "
                f"nonce: {activation_msg.nonce}, result: {x.shape} {x.dtype}"
            )

            # Prefetch next window
            next_window = self._next_local_layers(last_layer, self.prefetch_window_size)
            for nl in next_window:
                self._prefetch_to_ram(nl)

            # Evict layers not needed in next window
            for wl in window_layers:
                if wl not in next_window:
                    try:
                        self.weight_cache.evict_layer(wl)
                    except Exception:
                        pass

            # Optionally prefetch first next-window layer
            if next_window:
                self._enqueue_weight_prefetch(next_window[0])

            # Allocate output buffer
            output_pool_id = self.output_pool.allocate_for_layer(
                last_layer, cast(Tuple[int, ...], tuple(x.shape)), x.dtype
            )
            if output_pool_id is None:
                logger.error(f"Failed to allocate output buffer for layer {last_layer}")
                self.input_pool.release(activation_msg.pool_id)
                return

            output_buffer = self.output_pool.get_buffer(output_pool_id)
            if output_buffer is None:
                logger.error(f"Failed to get output buffer {output_pool_id}")
                return
            x_flat = x.flatten()
            output_buffer[: len(x_flat)] = x_flat

            # Create output activation message
            output_msg = ActivationMessage(
                nonce=activation_msg.nonce,
                layer_id=last_layer,
                pool_id=output_pool_id,
                shape=cast(Tuple[int, ...], tuple(x.shape)),
                batch_size=activation_msg.batch_size,
                timestamp=utc_epoch_now(),
                node_origin=f"node_{self.node_id}",
                dtype=str(x.dtype),
                callback_url=activation_msg.callback_url,
            )

            # Queue for sending
            try:
                self.activation_computed_queue.put(output_msg, timeout=10)
            except Exception as e:
                logger.error(f"Failed to queue computed activation for sending: {e}")
                self.output_pool.release(output_pool_id)

            # Clean up input resources
            self.input_pool.release(activation_msg.pool_id)

        except Exception as e:
            logger.exception(f"Error processing activation: {e}")

    async def _send_worker(self) -> None:
        """Async worker for sending activations to next node."""
        while self.running:
            try:
                # Check for computed activations to send
                try:
                    activation_msg = self.activation_computed_queue.get_nowait()
                    await self._send_activation(activation_msg)
                except Empty:
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Send worker error: {e}")
                await asyncio.sleep(0.05)

    async def _send_activation(self, activation_msg: ActivationMessage) -> None:
        """Send activation to next node.

        Args:
            activation_msg: Activation message to send
        """
        if not self._check_model_loaded() or not self.output_pool:
            logger.error(
                f"Node {self.node_id}: Cannot send activation - model not loaded"
            )
            return

        try:
            # Get data from output buffer
            output_buffer = self.output_pool.get_buffer(activation_msg.pool_id)
            if output_buffer is None:
                logger.error(f"Failed to get output buffer {activation_msg.pool_id}")
                return

            # Extract actual data
            t_ser = time.perf_counter()
            data_size = int(np.prod(activation_msg.shape))
            data = tensor_to_bytes(output_buffer.flatten()[:data_size])
            ser_ms = (time.perf_counter() - t_ser) * 1000.0

            # Send to next node or complete
            assert self.model_metadata is not None
            if (activation_msg.layer_id + 1) < self.model_metadata.num_layers:
                # More layers to process - forward to next shard
                if self.next_node_stub:
                    # Update per-hop timestamp
                    request = activation_msg.to_proto(data)
                    request.timestamp = utc_epoch_now()
                    t0 = time.perf_counter()
                    response = await self.next_node_stub.SendActivation(request)  # type: ignore
                    rpc_ms = (time.perf_counter() - t0) * 1000.0

                    if response.success:
                        logger.debug(
                            f"Successfully sent activation to {response.node_id}, "
                            f"nonce: {activation_msg.nonce}"
                        )
                    else:
                        logger.error(f"Failed to send activation: {response.message}")
                    logger.info(
                        f"[PROFILE][TX] node={self.node_id} nonce={activation_msg.nonce} "
                        f"next_layer={activation_msg.layer_id + 1} "
                        f"payload_kb={(len(data)/1024):.1f} serialize_ms={ser_ms:.3f} "
                        f"rpc_ms={rpc_ms:.2f}"
                    )
                else:
                    logger.error(
                        f"Cannot forward activation - no next node configured. "
                        f"Layer {activation_msg.layer_id + 1} needs processing but "
                        f"this is the final shard."
                    )
            else:
                # Final node - send to API
                logger.info(
                    f"FINAL OUTPUT - Nonce: {activation_msg.nonce}, "
                    f"Shape: {activation_msg.shape}, Layer: {activation_msg.layer_id}"
                )
                t_final_ser = time.perf_counter()
                serialized = tensor_to_bytes(output_buffer)
                final_ser_ms = (time.perf_counter() - t_final_ser) * 1000.0

                # Prefer gRPC callback
                if activation_msg.callback_url.startswith("grpc://"):
                    parsed = urlparse(activation_msg.callback_url)
                    addr = parsed.netloc  # host:port
                    if not addr:
                        logger.error(
                            f"Invalid gRPC callback URL: {activation_msg.callback_url}"
                        )
                    else:
                        if (self.api_channel is None) or (addr != self.api_address):
                            # Reconnect if address changed
                            if self.api_channel is not None:
                                try:
                                    await self.api_channel.close()
                                except Exception:
                                    pass
                            self.api_address = addr
                            self.api_channel = aio_grpc.insecure_channel(addr)
                            self.api_stub = ShardApiServiceStub(self.api_channel)

                        try:
                            t_rpc = time.perf_counter()
                            req = shard_api_comm_pb2.FinalActivationRequest(
                                nonce=activation_msg.nonce,
                                data=serialized,
                                batch_size=activation_msg.batch_size,
                                shape=list(activation_msg.shape),
                                dtype=str(output_buffer.dtype),
                                layer_id=activation_msg.layer_id,
                                timestamp=utc_epoch_now(),
                                node_origin=activation_msg.node_origin,
                            )
                            resp = await self.api_stub.SendFinalActivation(req)  # type: ignore
                            rpc_ms = (time.perf_counter() - t_rpc) * 1000.0
                            if not resp.success:
                                logger.error(
                                    f"API gRPC callback failed for {activation_msg.nonce}: "
                                    f"{resp.message}"
                                )
                            logger.info(
                                f"[PROFILE][TX-FINAL][gRPC] node={self.node_id} "
                                f"nonce={activation_msg.nonce} "
                                f"payload_kb={(len(serialized)/1024):.1f} "
                                f"serialize_ms={final_ser_ms:.3f} rpc_ms={rpc_ms:.2f}"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Error sending final activation via gRPC: {e}"
                            )
                else:
                    # Fallback to HTTP callback
                    t_rpc = time.perf_counter()
                    async with httpx.AsyncClient(
                        http1=False, http2=True, verify=False
                    ) as session:
                        await session.post(
                            activation_msg.callback_url,
                            json=RecieveResultRequest(
                                nonce=activation_msg.nonce,
                                batch_size=activation_msg.batch_size,
                                shape=activation_msg.shape,
                                dtype=str(output_buffer.dtype),
                                layer_id=activation_msg.layer_id,
                                timestamp=utc_epoch_now(),
                                node_origin=activation_msg.node_origin,
                                data=RecieveResultRequest.encode(serialized),
                            ).model_dump(),
                        )
                    rpc_ms = (time.perf_counter() - t_rpc) * 1000.0
                    logger.info(
                        f"[PROFILE][TX-FINAL][HTTP] node={self.node_id} "
                        f"nonce={activation_msg.nonce} "
                        f"payload_kb={(len(serialized)/1024):.1f} "
                        f"serialize_ms={final_ser_ms:.3f} rpc_ms={rpc_ms:.2f}"
                    )

            # Release output buffer
            self.output_pool.release(activation_msg.pool_id)

        except Exception as e:
            logger.exception(f"Error sending activation: {e}")

    async def shutdown(self) -> None:
        """Shutdown the node."""
        self.running = False

        # Stop HTTP server
        if self.http_server and not self.http_server.done():
            self.http_server.cancel()
            try:
                await self.http_server
            except asyncio.CancelledError:
                pass

        # Stop gRPC server
        if self.server:
            await self.server.stop(grace=5)

        # Close channels
        if self.next_node_channel:
            await self.next_node_channel.close()
        if self.api_channel:
            await self.api_channel.close()

        # Terminate compute thread
        if self.compute_thread:
            self.compute_thread.join(timeout=5)

        # Stop discovery service
        if self.discovery.is_running():
            logger.info(f"Stopping discovery service for node {self.node_id}")
            self.discovery.stop()
            self.discovery.free_instance()
        else:
            logger.warning(
                f"Discovery service for node {self.node_id} was not running"
            )

        # Stop background tasks
        for bgt in self.background_tasks:
            bgt.cancel()
        await asyncio.gather(*self.background_tasks, return_exceptions=True)

        logger.info(f"Node {self.node_id} shutdown complete")
