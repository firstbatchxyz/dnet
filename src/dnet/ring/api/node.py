"""Ring API node implementation with dynamic topology and model loading."""

import asyncio
import time
import uuid
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import mlx.core as mx
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from grpc import aio as aio_grpc
from hypercorn import Config
import hypercorn.asyncio as aio_hypercorn
from mlx_lm.tokenizer_utils import load_tokenizer

from dnet_p2p import DnetDeviceProperties, DnetP2P, discover_thunderbolt_connections
from dperf import profile_model
from dperf.profiler import ModelProfileSplit
from dsolver import DeviceProfile, halda_solve
from dsolver.gurobi_solver import HALDAResult

from ...protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from ...protos.shard_api_comm_pb2_grpc import (
    add_ShardApiServiceServicer_to_server,
)

from ...utils.logger import logger
from ...utils.model import ModelMetadata, get_model_metadata, load_api_layer_weights
from .utils import create_generate_step_for_ring_with_grpc
from ..api_models import (
    ChatBaseParams,
    ChatChoice,
    ChatCompletionReason,
    ChatLogProp,
    ChatMessage,
    ChatRequestModel,
    ChatResponseModel,
    CompletionRequestModel,
    DeviceInfo,
    LayerAssignment,
    LoadModelRequest,
    LoadModelResponse,
    PrepareTopologyRequest,
    PrepareTopologyResponse,
    RoleMapping,
    ShardLoadStatus,
)
from ..data_types import StopCondition
from ..model import get_ring_model
from .servicer import ApiServicer


async def arange(count: int):
    """Async range generator."""
    for i in range(count):
        yield i


async def azip(*async_iterables):
    """Async zip."""
    iterators = [aiter(it) for it in async_iterables]
    while True:
        try:
            results = await asyncio.gather(*[anext(it) for it in iterators])
            yield results
        except StopAsyncIteration:
            break


class RingApiNode:
    """API node for distributed inference ring with dynamic topology."""

    def __init__(
        self,
        http_port: int,
        grpc_port: int,
    ) -> None:
        """Initialize API node.

        Args:
            http_port: HTTP server port
            grpc_port: gRPC callback port
        """
        self.http_port = http_port
        self.grpc_port = grpc_port

        # Model state (loaded dynamically)
        self.model_metadata: Optional[ModelMetadata] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.generate_step: Optional[Any] = None
        self.model_name: Optional[str] = None

        # Topology state
        self.topology: Dict[str, Any] = {}
        self.layer_assignments: Dict[str, List[int]] = {}
        self.first_shard_address: Optional[str] = None

        # API
        self.app = FastAPI()
        self.running = False

        # gRPC
        self.http_server: Optional[Any] = None
        self.api_grpc_server: Optional[aio_grpc.Server] = None
        self.first_shard_channel: Optional[aio_grpc.Channel] = None
        self.first_shard_stub: Optional[DnetRingServiceStub] = None

        # Discovery
        self.discovery = DnetP2P("lib/dnet-p2p/lib")

        # Callback tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}

        logger.info(
            f"API node initialized on HTTP port {self.http_port}, "
            f"gRPC port {self.grpc_port}"
        )

    async def start(self, shutdown_trigger: Any = lambda: asyncio.Future()) -> None:
        """Start the API node.

        Args:
            shutdown_trigger: Shutdown trigger function
        """
        self.running = True

        # Start gRPC server for callbacks
        await self._start_grpc_server()

        # Start discovery
        self._start_discovery()

        # Start HTTP server
        await self._start_http_server(shutdown_trigger)

    def _start_discovery(self) -> None:
        """Start mDNS discovery service."""
        from secrets import token_hex
        from socket import gethostname

        hostname = gethostname()
        instance = f"api-{token_hex(4)}-{hostname}"
        self.discovery.create_instance(
            instance,
            hostname,
            "0.0.0.0",  # Bind to all addresses
            self.http_port,
            self.grpc_port,
            is_manager=True,  # API is a manager
        )
        self.discovery.start()
        logger.info("Discovery service started for API node")

    async def _start_grpc_server(self) -> None:
        """Start gRPC server for receiving callbacks from shards."""
        if self.api_grpc_server:
            return

        server = aio_grpc.server()
        servicer = ApiServicer(self)
        add_ShardApiServiceServicer_to_server(servicer, server)
        listen_addr = f"[::]:{self.grpc_port}"
        server.add_insecure_port(listen_addr)
        await server.start()
        self.api_grpc_server = server
        logger.info(f"API gRPC callback server started on {listen_addr}")

    async def _start_http_server(self, shutdown_trigger: Any) -> None:
        """Start HTTP server.

        Args:
            shutdown_trigger: Shutdown trigger function
        """
        await self._setup_routes()

        config = Config.from_mapping(
            bind=f"0.0.0.0:{self.http_port}",
            log_level="info",
            log_config=None,
            use_reloader=False,
            h2c=True,
        )

        await aio_hypercorn.serve(
            self.app,  # type: ignore # FIXME: !!!
            config,
            shutdown_trigger=shutdown_trigger,
        )

    async def _setup_routes(self) -> None:
        """Setup HTTP routes."""

        @self.app.exception_handler(ValueError)
        async def value_error_handler(request: Any, exc: ValueError):  # type: ignore
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": str(exc)},
            )

        @self.app.get("/ping")
        async def ping() -> JSONResponse:
            return JSONResponse(content={"status": "ok"})

        @self.app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse(
                content={
                    "status": "ok",
                    "model_loaded": self.model is not None,
                    "model_name": self.model_name,
                    "topology_configured": bool(self.topology),
                }
            )

        @self.app.get("/devices")
        async def get_devices() -> JSONResponse:
            """Get all discovered devices from mDNS."""
            devices = self.discovery.get_properties()
            devices_dict = {
                service_name: device_props.model_dump()
                for service_name, device_props in devices.items()
            }
            return JSONResponse(content={"devices": devices_dict})

        @self.app.post("/v1/prepare_topology")
        async def prepare_topology(
            req: PrepareTopologyRequest,
        ) -> PrepareTopologyResponse:  # type: ignore
            """Prepare topology for a model."""
            try:
                return await self._handle_prepare_topology(req)
            except Exception as e:
                logger.exception(f"Error in /v1/prepare_topology: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                )

        @self.app.post("/v1/load_model")
        async def load_model(req: LoadModelRequest) -> LoadModelResponse:  # type: ignore
            """Load model on shards with prepared topology."""
            try:
                return await self._handle_load_model(req)
            except Exception as e:
                logger.exception(f"Error in /v1/load_model: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                )

        @self.app.post("/v1/chat/completions")
        async def chat_completions(req: ChatRequestModel) -> ChatResponseModel:  # type: ignore
            """Handle chat completion requests."""
            if self.model is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No model loaded. Call /v1/load_model first.",
                )
            if not self.first_shard_stub:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Not connected to first shard",
                )
            return await self._handle_chat_completion(req)

        @self.app.post("/v1/completions")
        async def completions(req: CompletionRequestModel):  # type: ignore
            """Handle completion requests (not implemented)."""
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Endpoint not implemented",
            )

    async def _handle_prepare_topology(
        self, req: PrepareTopologyRequest
    ) -> PrepareTopologyResponse:
        """Handle topology preparation request.

        Args:
            req: Topology preparation request

        Returns:
            Topology preparation response
        """
        logger.info(f"Preparing topology for model: {req.model}")

        # Optionally force rediscovery
        if req.force_rediscover:
            # Give discovery time to refresh
            await asyncio.sleep(2.0)

        # Load model metadata
        model_metadata = get_model_metadata(req.model)

        # Calculate embedding size
        embedding_size = await self._get_embedding_size(model_metadata, req.model)

        # Profile model
        batch_sizes = [1, 2, 4, 8]
        sequence_length = 512
        model_profile = await self._profile_model(
            req.model, batch_sizes, sequence_length
        )

        # Get shards from discovery
        shards = self._get_shards_from_discovery()
        if not shards:
            raise ValueError("No shards discovered. Ensure shard nodes are running.")

        logger.info(f"Discovered {len(shards)} shards: {list(shards.keys())}")

        # Collect shard profiles
        shard_profiles = await self._collect_shard_profiles(
            shards, req.model, embedding_size, max_batch_exp=2
        )

        # Run solver
        device_names, solution = await self._run_solver(shard_profiles, model_profile)

        # Compute layer assignments, next service mapping, and prefetch windows
        layer_assignments, next_service_map, prefetch_windows = (
            self._compute_layer_assignments(device_names, solution, shards)
        )

        # Store topology
        self.topology = {
            "model": req.model,
            "num_layers": model_metadata.num_layers,
            "devices": device_names,
            "assignments": layer_assignments,
            "next_service_map": next_service_map,
            "prefetch_windows": prefetch_windows,
            "solution": asdict(solution),
        }

        # Build response
        devices_info = [
            DeviceInfo(
                service_name=name,
                local_ip=shards[name].local_ip,
                http_port=shards[name].server_port,
                grpc_port=shards[name].shard_port,
            )
            for name in device_names
        ]

        assignments_info = [
            LayerAssignment(
                service=name,
                layers=layer_assignments[name],
                next_service=next_service_map[name],
            )
            for name in device_names
        ]

        logger.info(
            f"Topology prepared: {len(device_names)} devices, {model_metadata.num_layers} layers"
        )

        return PrepareTopologyResponse(
            model=req.model,
            num_layers=model_metadata.num_layers,
            devices=devices_info,
            assignments=assignments_info,
            diagnostics={
                "solver_k": solution.k,
                "solver_objective": solution.obj_value,
                "solver_w": solution.w,
            },
        )

    async def _handle_load_model(self, req: LoadModelRequest) -> LoadModelResponse:
        """Handle load model request.

        Args:
            req: Load model request

        Returns:
            Load model response
        """
        logger.info(f"Loading model: {req.model}")
        start_time = time.perf_counter()

        # Create layer assignment map - flatten layers for internal use
        layer_assignments = {
            assignment.service: [
                layer for round_layers in assignment.layers for layer in round_layers
            ]
            for assignment in req.assignments
        }

        # Get prefetch windows from stored topology
        prefetch_windows = self.topology.get("prefetch_windows", {})
        if not prefetch_windows:
            logger.warning("No prefetch windows found in topology, using default of 1")
            prefetch_windows = {name: 1 for name in layer_assignments.keys()}

        # Get shards
        shards = self._get_shards_from_discovery()

        # Notify each shard to load their layers via HTTP
        shard_statuses: List[ShardLoadStatus] = []
        async with httpx.AsyncClient() as http_client:
            for assignment in req.assignments:
                service_name = assignment.service
                # Flatten layers for shard loading
                layers = [
                    layer for round_layers in assignment.layers for layer in round_layers
                ]

                if service_name not in shards:
                    logger.warning(f"Shard {service_name} not found in discovery")
                    shard_statuses.append(
                        ShardLoadStatus(
                            service_name=service_name,
                            success=False,
                            message="Shard not found in discovery",
                            layers_loaded=[],
                        )
                    )
                    continue

                shard_props = shards[service_name]

                # Get next node address from next_service in ring
                next_node_address = ""
                if assignment.next_service and assignment.next_service in shards:
                    next_shard = shards[assignment.next_service]
                    next_node_address = f"{next_shard.local_ip}:{next_shard.shard_port}"
                    logger.info(
                        f"Shard {service_name} next node in ring: {assignment.next_service} "
                        f"at {next_node_address}"
                    )
                else:
                    logger.info(
                        f"Shard {service_name} has no valid next service in ring"
                    )

                try:
                    # Get prefetch window for this shard
                    prefetch_window = prefetch_windows.get(service_name, 1)

                    # Get total layers from stored topology
                    total_layers = self.topology.get("num_layers", 0)

                    # Build API callback address (gRPC)
                    api_callback_address = f"0.0.0.0:{self.grpc_port}"

                    # Call load_model via HTTP
                    url = f"http://{shard_props.local_ip}:{shard_props.server_port}/load_model"
                    payload = {
                        "model_path": req.model,
                        "layers": layers,
                        "warmup": True,
                        "next_node_address": next_node_address,
                        "prefetch_window": prefetch_window,
                        "total_layers": total_layers,
                        "api_callback_address": api_callback_address,
                    }

                    response = await http_client.post(url, json=payload, timeout=300.0)
                    result = response.json()

                    shard_statuses.append(
                        ShardLoadStatus(
                            service_name=service_name,
                            success=result.get("success", False),
                            message=result.get("message", ""),
                            layers_loaded=result.get("layers_loaded", []),
                        )
                    )

                    logger.info(
                        f"Shard {service_name} load result: success={result.get('success')}, "
                        f"message={result.get('message')}"
                    )

                except Exception as e:
                    logger.exception(
                        f"Error loading model on shard {service_name}: {e}"
                    )
                    shard_statuses.append(
                        ShardLoadStatus(
                            service_name=service_name,
                            success=False,
                            message=f"Error: {str(e)}",
                            layers_loaded=[],
                        )
                    )

        # Check if all shards loaded successfully
        all_success = all(status.success for status in shard_statuses)

        if not all_success:
            failed_shards = [
                status.service_name for status in shard_statuses if not status.success
            ]
            logger.error(f"Failed to load model on shards: {failed_shards}")

        # If successful, load API-side model components
        if all_success:
            try:
                self.model_metadata = get_model_metadata(req.model)
                self.model_name = req.model

                # Load tokenizer
                self.tokenizer = load_tokenizer(Path(req.model), {})

                # Load API-side model (embeddings, lm_head, etc.)
                self.model = get_ring_model(
                    self.model_metadata.model_type,
                    self.model_metadata.model_config,
                    is_api_layer=True,
                )
                load_api_layer_weights(self.model_metadata, self.model)

                # Store topology
                self.layer_assignments = layer_assignments

                # Connect to first shard (head device)
                # Find the shard with layer 0
                first_shard_name = None
                for name, layers in layer_assignments.items():
                    if 0 in layers:
                        first_shard_name = name
                        break

                if first_shard_name and first_shard_name in shards:
                    shard_props = shards[first_shard_name]
                    self.first_shard_address = (
                        f"{shard_props.local_ip}:{shard_props.shard_port}"
                    )
                    await self._connect_first_shard()
                else:
                    logger.warning("Could not identify first shard (with layer 0)")

                logger.info(f"API-side model loaded successfully for {req.model}")

            except Exception as e:
                logger.exception(f"Error loading API-side model: {e}")
                all_success = False
                shard_statuses.append(
                    ShardLoadStatus(
                        service_name="api",
                        success=False,
                        message=f"API model load error: {str(e)}",
                        layers_loaded=[],
                    )
                )

        total_load_time_ms = (time.perf_counter() - start_time) * 1000.0

        return LoadModelResponse(
            model=req.model,
            success=all_success,
            shard_statuses=shard_statuses,
            total_load_time_ms=total_load_time_ms,
        )

    async def _connect_first_shard(self) -> bool:
        """Connect to first shard in ring.

        Returns:
            True if connected, False otherwise
        """
        if not self.first_shard_address:
            return False

        if self.first_shard_channel:
            return True

        try:
            self.first_shard_channel = aio_grpc.insecure_channel(
                self.first_shard_address
            )
            self.first_shard_stub = DnetRingServiceStub(self.first_shard_channel)

            # Prepare generate_step with gRPC callback
            callback_addr = f"0.0.0.0:{self.grpc_port}"
            self.generate_step = create_generate_step_for_ring_with_grpc(
                self.first_shard_stub,
                callback_protocol="grpc",
                callback_addr=callback_addr,
            )

            logger.info(f"Connected to first shard at {self.first_shard_address}")
            return True

        except Exception as e:
            logger.warning(
                f"Failed to connect to first shard {self.first_shard_address}: {e}"
            )
            self.first_shard_channel = None
            self.first_shard_stub = None
            self.generate_step = None
            return False

    def _get_shards_from_discovery(self) -> Dict[str, DnetDeviceProperties]:
        """Get shards from discovery (excluding manager nodes).

        Returns:
            Dictionary of shard service names to properties
        """
        devices = self.discovery.get_properties()
        return {k: v for k, v in devices.items() if not v.is_manager}

    async def _get_embedding_size(
        self, model_metadata: ModelMetadata, model_name: str
    ) -> int:
        """Get embedding size from model metadata.

        Args:
            model_metadata: Model metadata
            model_name: Model name (for logging)

        Returns:
            Embedding size
        """
        # Try to get embedding_size first, fallback to hidden_size
        embedding_size = model_metadata.model_config.get("embedding_size")
        if embedding_size is None:
            # Try to infer from embed_tokens tensor dimensions
            if model_metadata.embed_tokens and "weight" in model_metadata.embed_tokens:
                embedding_size = model_metadata.embed_tokens["weight"].shape[1]
            else:
                # Fallback to hidden_size
                embedding_size = model_metadata.model_config.get("hidden_size")

        if embedding_size is None:
            raise ValueError(
                f"Could not find embedding_size or hidden_size in model config for {model_name}"
            )

        return embedding_size

    async def _profile_model(
        self, repo_id: str, batch_sizes: List[int], sequence_length: int
    ) -> ModelProfileSplit:
        """Profile model using dperf.

        Args:
            repo_id: Hugging Face repository ID
            batch_sizes: List of batch sizes to profile
            sequence_length: Sequence length to profile

        Returns:
            Model profile
        """
        model_profile: ModelProfileSplit = profile_model(
            repo_id=repo_id,
            batch_sizes=batch_sizes,
            sequence_length=sequence_length,
        )
        logger.info(f"Model profiling completed: L = {model_profile.L}")
        return model_profile

    async def _collect_shard_profiles(
        self,
        shards: Dict[str, DnetDeviceProperties],
        repo_id: str,
        embedding_size: int,
        max_batch_exp: int = 1,  # very small default for speed
    ) -> Dict[str, Any]:
        """Collect profile data from all shards.

        Args:
            shards: Discovered shards
            repo_id: Model repository ID
            embedding_size: Model embedding size
            max_batch_exp: Maximum batch size exponent

        Returns:
            Collected shard profiles
        """
        # Calculate payload sizes
        base_size = embedding_size * 4  # float32
        payload_sizes = [base_size * batch_size for batch_size in [1, 2, 4, 8]]

        this_device = self.discovery.get_own_properties()

        logger.info(
            f"Model {repo_id}: embedding_size={embedding_size}, "
            f"payload_sizes={payload_sizes}"
        )

        # Find Thunderbolt connections
        thunderbolt_conns = discover_thunderbolt_connections(shards)

        # Call each shard's /profile endpoint
        shard_profiles: Dict[str, Any] = {}
        async with httpx.AsyncClient() as client:
            for service_name, shard_props in shards.items():
                if shard_props.is_manager:
                    logger.warning(
                        f"Skipping manager node {service_name} in profile collection"
                    )
                    continue

                server_port, server_ip = shard_props.server_port, shard_props.local_ip

                # Serialize Thunderbolt connections for this shard
                thunderbolt_for_shard = thunderbolt_conns.get(service_name, {})
                thunderbolt_conns_json = {
                    k: {"ip": v[0], "instance": v[1].model_dump()}
                    for k, v in thunderbolt_for_shard.items()
                }

                try:
                    url = f"http://{server_ip}:{server_port}/profile"
                    logger.info(
                        f"Calling /profile endpoint for shard {service_name} at {url}"
                    )

                    response = await client.post(
                        url,
                        json={
                            "devices": {
                                name: props.model_dump(
                                    include={
                                        "local_ip",
                                        "server_port",
                                        "shard_port",
                                        "is_manager",
                                    }
                                )
                                for name, props in shards.items()
                            },
                            "payload_sizes": payload_sizes,
                            "thunderbolts": thunderbolt_conns_json,
                            "max_batch_exp": max_batch_exp,
                            "repo_id": repo_id,
                        },
                        timeout=1000.0,
                    )

                    if response.status_code == 200:
                        profile_data = response.json()
                        logger.info(
                            f"Successfully collected profile from {service_name}"
                        )

                        # Mark head device (same local IP as API)
                        if shard_props.local_ip == this_device.local_ip:
                            profile_data["profile"]["is_head"] = True

                        shard_profiles[service_name] = profile_data
                    else:
                        logger.error(
                            f"Failed to get profile from {service_name}: "
                            f"{response.status_code}"
                        )

                except Exception as e:
                    logger.exception(f"Error calling /profile for {service_name}: {e}")

        logger.info(f"Collected profiles from {len(shard_profiles)} shards")
        return shard_profiles

    async def _run_solver(
        self, shard_profiles: Dict[str, Any], model_profile_split: ModelProfileSplit
    ) -> Tuple[List[str], HALDAResult]:
        """Run dsolver with model and device profiles.

        Args:
            shard_profiles: Collected shard profiles
            model_profile_split: Model profile from dperf

        Returns:
            Tuple of (device names in solver order, solver result)
        """
        from dsolver.components.gurobi_loader import (
            load_device_profile_from_dict,
            load_model_profile_from_dict,
        )

        # Convert shard profiles to device profiles
        device_profiles: List[DeviceProfile] = []
        device_names: List[str] = []
        for shard_name, profile_data in shard_profiles.items():
            device_profile_data = profile_data["profile"]
            device_profile = load_device_profile_from_dict(device_profile_data)
            device_profiles.append(device_profile)
            device_names.append(shard_name)

        # Sort so head device is first
        device_profiles.sort(key=lambda x: x.is_head, reverse=True)

        if not device_profiles:
            raise ValueError("No valid device profiles found")

        logger.info(f"Running solver with {len(device_profiles)} device profiles")

        model_profile = load_model_profile_from_dict(asdict(model_profile_split))
        solution = halda_solve(
            device_profiles,
            model_profile,
            time_limit_per_k=5.0,
            mip_gap=1e-4,
            max_outer_iters=50,
            plot=False,
        )

        logger.info(f"Solver completed: k={solution.k}, objective={solution.obj_value}")

        return (device_names, solution)

    def _compute_layer_assignments(
        self,
        device_names: List[str],
        solution: HALDAResult,
        shards: Dict[str, DnetDeviceProperties],
    ) -> Tuple[Dict[str, List[List[int]]], Dict[str, Optional[str]], Dict[str, int]]:
        """Compute round-aware layer assignments, next node mapping, and prefetch windows from solver output.

        Args:
            device_names: Device names in solver order
            solution: Solver result
            shards: Discovered shards

        Returns:
            Tuple of (layer assignments per device per round, next service per device in ring, prefetch window per device)
        """
        if len(solution.w) != len(shards) or len(device_names) != len(shards):
            raise ValueError(
                f"Device count mismatch: solution={len(solution.w)}, "
                f"shards={len(shards)}"
            )

        num_layers = sum(solution.w) * solution.k
        logger.info(f"Distributing {num_layers} layers to {len(shards)} devices in {solution.k} rounds")

        # Assign layers in round-robin fashion, grouped by rounds
        # Each device gets k sublists (one per round)
        layer_assignments: Dict[str, List[List[int]]] = {name: [[] for _ in range(solution.k)] for name in device_names}
        current_layer = 0

        for round_idx in range(solution.k):
            for device_idx, device_name in enumerate(device_names):
                for _ in range(solution.w[device_idx]):
                    layer_assignments[device_name][round_idx].append(current_layer)
                    current_layer += 1

        assert current_layer == num_layers, (
            f"Assigned {current_layer} layers, expected {num_layers}"
        )

        # Compute next service for each device in ring topology
        # In ring: dev1 -> dev2 -> ... -> devN -> dev1 (wraps around)
        # Each shard will detect when processing the final layer and send to API
        next_service_map: Dict[str, Optional[str]] = {}

        if len(device_names) == 1:
            # Single device: forwards to itself in a loop
            next_service_map[device_names[0]] = device_names[0]
            logger.info(f"Ring (single device): {device_names[0]} -> SELF (loops back)")
        else:
            # Multiple devices: each forwards to the next in the ring
            for i, service_name in enumerate(device_names):
                if i < len(device_names) - 1:
                    # Forward to next device
                    next_service_map[service_name] = device_names[i + 1]
                else:
                    # Last device wraps to first device
                    next_service_map[service_name] = device_names[0]

            # Log ring topology
            for service_name in device_names:
                logger.info(
                    f"Ring: {service_name} -> {next_service_map[service_name]}"
                )

        # Compute prefetch window for each device: total_layers_per_device / k
        prefetch_windows: Dict[str, int] = {}
        for service_name, rounds_layers in layer_assignments.items():
            # Flatten to count total layers
            total_layers = sum(len(round_layers) for round_layers in rounds_layers)
            if total_layers > 0:
                prefetch_window = max(1, total_layers // solution.k)
                prefetch_windows[service_name] = prefetch_window
                logger.info(
                    f"Prefetch window for {service_name}: {prefetch_window} "
                    f"(total_layers={total_layers}, k={solution.k})"
                )
            else:
                prefetch_windows[service_name] = 1

        logger.info(f"Layer assignments (by rounds): {layer_assignments}")
        return layer_assignments, next_service_map, prefetch_windows

    async def _handle_chat_completion(self, req: ChatRequestModel) -> ChatResponseModel:
        """Handle chat completion request.

        Args:
            req: Chat request

        Returns:
            Chat response
        """
        stop_id_sequences: List[List[int]] = [
            self.tokenizer.encode(stop_word, add_special_tokens=False)  # type: ignore
            for stop_word in req.stop  # type: ignore
        ]
        prompt = await self._convert_chat(req.messages)
        prompt_array = mx.array(self.tokenizer.encode(prompt))  # type: ignore
        return await self._handle_completion(req, prompt_array, stop_id_sequences)

    async def _convert_chat(
        self,
        messages: List[ChatMessage],
        role_mapping: RoleMapping = RoleMapping(),
    ) -> str:
        """Convert chat messages to prompt string.

        Args:
            messages: Chat messages
            role_mapping: Role mapping for fallback

        Returns:
            Formatted prompt string
        """
        # Use tokenizer's chat template if available
        if (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None  # type: ignore
        ):
            message_dicts = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(  # type: ignore
                    message_dicts,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                logger.info(
                    f"Using tokenizer chat template, prompt preview: {prompt[:100]}..."
                )
                return prompt
            except Exception as e:
                logger.warning(
                    f"Failed to apply chat template: {e}, falling back to default"
                )

        # Fallback to default role mapping
        buf = StringIO()
        for msg in messages:
            role_prefix = getattr(role_mapping, msg.role, "")
            buf.write(f"{role_prefix}{msg.content}{role_mapping.stop}")
        return buf.getvalue().rstrip() + role_mapping.assistant

    async def _handle_completion(
        self,
        req: ChatRequestModel,
        prompt: mx.array,
        stop_id_sequences: List[List[int]],
    ) -> ChatResponseModel:
        """Handle completion generation.

        Args:
            req: Chat request
            prompt: Tokenized prompt
            stop_id_sequences: Stop sequences as token IDs

        Returns:
            Chat response
        """
        profile_enabled = bool(getattr(req, "profile", False))
        t_start = time.perf_counter()
        t_first_token = None
        nonce = f"chatcmpl-{uuid.uuid4()}"
        detokenizer = self.tokenizer.detokenizer  # type: ignore
        detokenizer.reset()
        tokens: List[int] = []
        completion_reason = ChatCompletionReason.LENGTH
        stop_sequence_suffix = None
        token_logprobs: List[float] = []
        top_tokens: List[Dict[int, float]] = []

        async for (token, logprobs), _ in azip(
            self.generate_step(
                nonce=nonce,
                node_origin=f"0.0.0.0:{self.http_port}",
                prompt=prompt,
                model=self.model,
                pending_requests=self.pending_requests,
                params=ChatBaseParams(
                    model=req.model,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                    repetition_context_size=req.repetition_context_size,
                    logit_bias=req.logit_bias,
                ),
            ),  # type: ignore
            arange(req.max_tokens or 0),
        ):
            if profile_enabled and t_first_token is None:
                t_first_token = time.perf_counter()
            detokenizer.add_token(token)
            tokens.append(token)

            if req.logprobs and (req.logprobs > 0):
                sorted_indices = mx.argpartition(-logprobs, kth=req.logprobs - 1)
                top_indices = sorted_indices[: (req.logprobs or 0)]
                top_logprobs_array = logprobs[top_indices]
                top_token_info = zip(
                    top_indices.tolist(),  # type: ignore # FIXME: !!!
                    top_logprobs_array.tolist(),  # type: ignore # FIXME: !!!
                )
                top_tokens.append(dict(top_token_info))

            token_logprobs.append(logprobs[token].item())

            stop_condition = await self._stopping_criteria(
                tokens,
                stop_id_sequences,
                self.tokenizer.eos_token_id,  # type: ignore
            )
            if stop_condition.stop_met:
                completion_reason = ChatCompletionReason.STOP
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(  # type: ignore
                        tokens[-stop_condition.trim_length :]
                    )
                break

        detokenizer.finalize()
        text = (
            detokenizer.text
            if stop_sequence_suffix is None
            else detokenizer.text[: -len(stop_sequence_suffix)]
        )

        # Build optional metrics
        metrics = None
        if profile_enabled:
            t_end = time.perf_counter()
            total_s = max(t_end - t_start, 1e-9)
            ttfb_ms = (t_first_token - t_start) * 1000.0 if t_first_token else None
            gen_s = max((t_end - (t_first_token or t_start)), 1e-9)
            tokens_generated = len(tokens)
            metrics = {
                "total_ms": round(total_s * 1000.0, 3),
                "ttfb_ms": round(ttfb_ms, 3) if ttfb_ms is not None else None,
                "token_gen_ms": round(gen_s * 1000.0, 3),
                "tokens_generated": tokens_generated,
                "tps_overall": round(tokens_generated / total_s, 4)
                if tokens_generated
                else 0.0,
                "tps_decoding": round(tokens_generated / gen_s, 4)
                if tokens_generated
                else 0.0,
            }

        return await self._generate_response(
            nonce,
            text,
            completion_reason,
            len(prompt),
            len(tokens),
            token_logprobs=token_logprobs,
            top_logprobs=top_tokens,
            tokens=tokens,
            metrics=metrics,
        )

    async def _stopping_criteria(
        self,
        tokens: List[int],
        stop_id_sequences: List[List[int]],
        eos_token_id: Union[int, None],
    ) -> StopCondition:
        """Check if generation should stop.

        Args:
            tokens: Generated tokens so far
            stop_id_sequences: Stop sequences as token IDs
            eos_token_id: EOS token ID

        Returns:
            Stop condition
        """
        if tokens and tokens[-1] == eos_token_id:
            return StopCondition(stop_met=True, trim_length=1)
        return next(
            (
                StopCondition(stop_met=True, trim_length=len(stop_ids))
                for stop_ids in stop_id_sequences
                if tokens[-len(stop_ids) :] == stop_ids
            ),
            StopCondition(stop_met=False, trim_length=0),
        )

    async def _generate_response(
        self,
        nonce: str,
        text: str,
        finish_reason: Optional[ChatCompletionReason] = None,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None,
        top_logprobs: Optional[List[Dict[int, float]]] = None,
        tokens: Optional[List[int]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ChatResponseModel:
        """Generate chat response.

        Args:
            nonce: Request nonce
            text: Generated text
            finish_reason: Completion reason
            prompt_token_count: Number of prompt tokens
            completion_token_count: Number of completion tokens
            token_logprobs: Token log probabilities
            top_logprobs: Top token log probabilities
            tokens: Generated tokens
            metrics: Performance metrics

        Returns:
            Chat response
        """
        return ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason=finish_reason,
                    message=ChatMessage(role="assistant", content=text),
                    logprop=ChatLogProp(
                        token_logprobs=token_logprobs or [],
                        top_logprobs=top_logprobs or [],
                        tokens=tokens,
                    ),
                )
            ],
            usage={
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": (prompt_token_count or 0)
                + (completion_token_count or 0),
            },
            metrics=metrics,
        )

    async def shutdown(self) -> None:
        """Shutdown the node."""
        self.running = False

        # Stop HTTP server
        if self.http_server:
            self.http_server.should_exit = True
            await asyncio.sleep(1)

        # Close channels
        if self.first_shard_channel:
            await self.first_shard_channel.close()

        # Stop gRPC server
        if self.api_grpc_server:
            await self.api_grpc_server.stop(grace=5)

        # Stop discovery
        if self.discovery.is_running():
            logger.info("Stopping discovery service for API node")
            self.discovery.stop()
            self.discovery.free_instance()
        else:
            logger.warning("Discovery service was not running")

        logger.info("API server shutdown complete")
