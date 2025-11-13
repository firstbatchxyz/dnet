"""Ring API node implementation with dynamic topology and model loading."""

import asyncio
import time
import uuid
import json
from io import StringIO
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx
import mlx.core as mx
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from grpc import aio as aio_grpc
import hypercorn.asyncio as aio_hypercorn
from mlx_lm.tokenizer_utils import load_tokenizer

from dnet_p2p import (
    DnetDeviceProperties,
    AsyncDnetP2P,
    discover_all_thunderbolt_connections,
    ThunderboltConnection,
)
from distilp.solver import halda_solve, HALDAResult
from distilp.profiler import profile_model
from distilp.common import DeviceProfile, ModelProfile

from ..observability import load_settings

from ...protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from ...protos.shard_api_comm_pb2_grpc import (
    add_ShardApiServiceServicer_to_server,
)

from ...utils.logger import logger
from ...utils.banner import print_startup_banner
from ...utils.latency import LatencyResults, calculate_median_latency_seconds
from ...utils.model import (
    ModelMetadata,
    get_model_config_json,
    resolve_tokenizer_dir,
)
from .utils import (
    create_generate_step_for_ring_with_grpc,
    compute_layer_assignments,
    optimize_device_ordering,
    postprocess_single_round,
)
from .models import (
    ChatParams,
    ChatUsage,
    ChatChoice,
    ChatCompletionReason,
    ChatLogProbs,
    ChatMessage,
    ChatRequestModel,
    ChatResponseModel,
    CompletionRequestModel,
    APILoadModelRequest,
    APILoadModelResponse,
    PrepareTopologyManualRequest,
    PrepareTopologyRequest,
    RoleMapping,
    ShardLoadStatus,
    ShardUnloadStatus,
    UnloadModelResponse,
)
from ..shard.models import (
    MeasureLatencyRequest,
    MeasureLatencyResponse,
    ShardProfileRequest,
    ShardLoadModelRequest,
    ShardLoadModelResponse,
    ShardProfileResponse,
)
from ..data_types import StopCondition
from .servicer import ShardApiServicer
from ..common import TopologyInfo, LayerAssignment


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
        compression_pct: float = 0.0,
    ) -> None:
        """Initialize API node.

        Args:
            http_port: HTTP server port
            grpc_port: gRPC callback port
        """
        self.http_port = http_port
        self.grpc_port = grpc_port

        self.model_metadata: Optional[ModelMetadata] = None
        self.tokenizer: Optional[Any] = None
        self.generate_step: Optional[Any] = None
        self.topology: Optional[TopologyInfo] = None
        self.app = FastAPI()
        self.running = False
        try:
            self._compression_pct = max(0.0, min(100.0, float(compression_pct)))
        except ValueError:
            self._compression_pct = 0.0
        self.http_server: Optional[Any] = None
        self.api_grpc_server: Optional[aio_grpc.Server] = None
        self.first_shard_channel: Optional[aio_grpc.Channel] = None
        self.first_shard_stub: Optional[DnetRingServiceStub] = None
        self.discovery = AsyncDnetP2P("lib/dnet-p2p/lib")
        self.pending_requests: Dict[str, asyncio.Future] = {}

        # Print ASCII art banner if available
        try:
            print_startup_banner()
        except Exception:
            pass

        logger.info(
            "API node initialized on HTTP port %s, gRPC port %s",
            self.http_port,
            self.grpc_port,
        )

    async def start(self, shutdown_trigger: Any = lambda: asyncio.Future()) -> None:
        """Start the API node.

        Args:
            shutdown_trigger: Shutdown trigger function
        """
        self.running = True

        await self._start_grpc_server()
        await self._start_discovery()
        await self._start_http_server(shutdown_trigger)
        await asyncio.sleep(0.2)

    async def _start_discovery(self) -> None:
        """Start discovery service."""
        from secrets import token_hex
        from socket import gethostname

        hostname = gethostname()
        instance = f"api-{token_hex(4)}-{hostname}"
        self.discovery.create_instance(
            instance,
            self.http_port,
            self.grpc_port,
            is_manager=True,
        )
        await self.discovery.async_start()
        logger.info("Discovery service started for API node")

    async def _start_grpc_server(self) -> None:
        """Start gRPC server for receiving callbacks from shards."""
        if self.api_grpc_server:
            return

        server = aio_grpc.server()
        servicer = ShardApiServicer(self)  # type: ignore # FIXME: !!!
        add_ShardApiServiceServicer_to_server(servicer, server)
        listen_addr = f"[::]:{self.grpc_port}"
        server.add_insecure_port(listen_addr)
        await server.start()
        self.api_grpc_server = server
        logger.info("API gRPC callback server started on %s", listen_addr)

    async def _start_http_server(self, shutdown_trigger: Any) -> None:
        """Start HTTP server.

        Args:
            shutdown_trigger: Shutdown trigger function
        """
        from hypercorn import Config

        await self._setup_routes()

        config = Config.from_mapping(
            bind=f"0.0.0.0:{self.http_port}",
            log_level="info",
            log_config=None,
            use_reloader=False,
            h2c=True,
        )

        self.http_server = asyncio.create_task(
            aio_hypercorn.serve(
                self.app,  # type: ignore
                config,
                shutdown_trigger=shutdown_trigger,
            )
        )
        logger.info("API HTTP server started on port %s", self.http_port)

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
            model_loaded = (self.tokenizer is not None) and (
                self.generate_step is not None
            )
            return JSONResponse(
                content={
                    "status": "ok",
                    "model": self.topology.model if self.topology else None,
                    "model_loaded": model_loaded,
                    "topology_configured": bool(self.topology),
                }
            )

        @self.app.get("/v1/topology")
        async def topology() -> TopologyInfo:
            if self.topology:
                return self.topology
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No topology configured. Call /v1/prepare_topology first.",
                )

        @self.app.get("/v1/devices")
        async def get_devices() -> JSONResponse:
            devices = await self.discovery.async_get_properties()
            devices_dict = {
                instance: device_props.model_dump()
                for instance, device_props in devices.items()
            }
            return JSONResponse(content={"devices": devices_dict})

        @self.app.post("/v1/prepare_topology")
        async def prepare_topology(
            req: PrepareTopologyRequest,
        ) -> TopologyInfo:
            try:
                return await self._handle_prepare_topology(req)
            except Exception as e:
                logger.exception("Error in /v1/prepare_topology: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                ) from e

        @self.app.post("/v1/prepare_topology_manual")
        async def prepare_topology_manual(
            req: PrepareTopologyManualRequest,
        ) -> TopologyInfo:
            try:
                return await self._handle_prepare_topology_manual(req)

            except Exception as e:
                logger.exception("Error in /v1/prepare_topology_manual: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                ) from e

        @self.app.post("/v1/load_model")
        async def load_model(req: APILoadModelRequest) -> APILoadModelResponse:
            try:
                return await self._handle_load_model(req)
            except Exception as e:
                logger.exception("Error in /v1/load_model: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                ) from e

        @self.app.post("/v1/unload_model")
        async def unload_model() -> UnloadModelResponse:
            try:
                return await self._handle_unload_model()
            except Exception as e:
                logger.exception("Error in /v1/unload_model: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                ) from e

        @self.app.post("/v1/cleanup_repacked")
        async def cleanup_repacked(body: Dict[str, Any] | None = None) -> JSONResponse:  # type: ignore
            """Ask all shards to delete repacked per-layer weights to free disk.

            Body JSON (all fields optional):
              - model_id: restrict cleanup to this model bucket
              - all: when true, remove the entire repack directory base
            """
            payload = body or {}
            shards = await self._get_shards_from_discovery()
            results: Dict[str, Any] = {}
            async with httpx.AsyncClient() as http_client:
                for name, props in shards.items():
                    url = (
                        f"http://{props.local_ip}:{props.server_port}/cleanup_repacked"
                    )
                    try:
                        resp = await http_client.post(url, json=payload, timeout=30.0)
                        results[name] = resp.json()
                    except Exception as e:
                        results[name] = {"error": str(e)}
            return JSONResponse(content={"results": results})

        @self.app.post("/v1/chat/completions")
        async def chat_completions(
            req: ChatRequestModel,
        ) -> ChatResponseModel:
            """Handle chat completion requests.

            If streaming is requested, returns a StreamingResponse."""
            if (self.tokenizer is None) or (self.generate_step is None):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No model loaded. Call /v1/load_model first.",
                )
            if not self.first_shard_stub:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Not connected to first shard",
                )
            if req.stream:
                # FIXME: return type mismatch here
                return StreamingResponse(
                    self._stream_chat(req), media_type="text/event-stream"
                )  # type: ignore
            else:
                return await self._handle_chat_completion(req)

        @self.app.post("/v1/completions")
        async def completions(req: CompletionRequestModel):  # type: ignore
            if req.stream:
                return StreamingResponse(
                    self._stream_completion(req), media_type="text/event-stream"
                )
            return await self._handle_text_completion(req)

    async def _handle_prepare_topology(
        self, req: PrepareTopologyRequest
    ) -> TopologyInfo:
        """Handle topology preparation request.

        Args:
            req: Topology preparation request

        Returns:
            Topology preparation response
        """
        logger.info("Preparing topology for model: %s", req.model)

        # Load only config.json to avoid weight downloads on API node
        cfg = get_model_config_json(req.model)

        if (
            str(cfg.get("model_type", "")).strip().lower() == "gpt_oss"
            and req.kv_bits != "fp16"
        ):
            raise ValueError("GPT-OSS models only support kv_bits='fp16'")

        num_layers_raw = cfg.get("num_hidden_layers")
        if not isinstance(num_layers_raw, int):
            raise ValueError(
                "num_hidden_layers missing or invalid in config.json; cannot prepare topology without scanning weights"
            )
        num_layers = num_layers_raw

        embedding_raw = cfg.get("embedding_size")
        if isinstance(embedding_raw, int):
            embedding_size = embedding_raw
        else:
            hidden_raw = cfg.get("hidden_size")
            if not isinstance(hidden_raw, int):
                raise ValueError(
                    "embedding_size/hidden_size missing or invalid in config.json; cannot profile payload size"
                )
            embedding_size = hidden_raw

        # Profile model
        batch_sizes = [1]
        model_profile = await self._profile_model(req.model, batch_sizes, req.seq_len)

        # Get shards from discovery
        shards = await self._get_shards_from_discovery()
        if not shards:
            raise ValueError("No shards discovered. Ensure shard nodes are running.")

        logger.info("Discovered %d shards: %s", len(shards), list(shards.keys()))

        thunderbolt_conns = discover_all_thunderbolt_connections(shards)
        shard_profiles = await self._collect_shard_profiles(
            shards,
            req.model,
            embedding_size,
            req.max_batch_exp,
            batch_sizes,
            thunderbolt_conns,
        )
        optimized_device_name_order = optimize_device_ordering(
            shard_profiles, thunderbolt_conns
        )
        solution = await self._run_solver(
            shard_profiles, model_profile, optimized_device_name_order, req.kv_bits
        )
        optimized_device_name_order, solution = postprocess_single_round(
            optimized_device_name_order, solution
        )
        layer_assignments = compute_layer_assignments(
            optimized_device_name_order,
            shards,
            solution.w,
            solution.n,
            solution.k,
        )

        shards_list = [shards[name] for name in optimized_device_name_order]
        self.topology = TopologyInfo(
            model=req.model,
            kv_bits=req.kv_bits,
            num_layers=num_layers,
            devices=shards_list,
            assignments=layer_assignments,
            solution=solution,
        )

        # Optional, detailed solver print when profiling is enabled
        try:
            obs = load_settings()
            if obs.enabled:
                dev_order = ", ".join(optimized_device_name_order)
                logger.info(
                    "[SOLUTION] k=%s; devices=[%s]; w=%s; n=%s; obj=%s; total_layers=%s",
                    solution.k,
                    dev_order,
                    list(solution.w),
                    list(solution.n),
                    solution.obj_value,
                    num_layers,
                )
        except Exception:
            pass
        print(
            f"Topology solution: k {solution.k}, w {solution.w}, n {solution.n}, objective: {solution.obj_value}"
        )

        logger.info(
            "Topology prepared: %d devices, %d layers",
            len(shards_list),
            num_layers,
        )

        return self.topology

    async def _handle_prepare_topology_manual(
        self, req: PrepareTopologyManualRequest
    ) -> TopologyInfo:
        """Handle manual topology preparation without discovery."""
        logger.info("Preparing manual topology for model: %s", req.model)

        device_names = [d.instance for d in req.devices]
        if len(device_names) != len(set(device_names)):
            raise ValueError("Device names must be unique in manual topology")

        # FIXME: may not need normalized array here, just use assignments
        device_names = set(device_names)
        normalized: List[LayerAssignment] = []
        for assignment in req.assignments:
            if assignment.instance not in device_names:
                raise ValueError(
                    f"Assignment references unknown device: {assignment.instance}"
                )
            normalized.append(
                LayerAssignment(
                    instance=assignment.instance,
                    layers=assignment.layers,
                    next_instance=assignment.next_instance,
                    window_size=assignment.window_size,
                    residency_size=assignment.window_size,  # use window_size as residency_size for manual topology
                )
            )

        num_layers = req.num_layers
        if num_layers is None:
            flat = [layer for aa in normalized for rr in aa.layers for layer in rr]
            if not flat:
                raise ValueError("No layers provided in assignments")
            num_layers = max(flat) + 1

        # FIXME: we can perhaps use req.devices as is
        devices_props: List[DnetDeviceProperties] = []
        for d in req.devices:
            devices_props.append(
                DnetDeviceProperties(
                    is_manager=False,
                    is_busy=False,
                    instance=d.instance,
                    server_port=d.server_port,
                    shard_port=d.shard_port,
                    local_ip=d.local_ip,
                )
            )

        # FIXME: may not need this edge case at all, probably redundant
        if any(a.next_instance is None for a in normalized) and len(normalized) > 1:
            order = sorted(
                normalized,
                key=lambda aa: min([layer for rr in aa.layers for layer in rr])
                if aa.layers
                else (1 << 30),
            )
            ring_map = {
                order[i].instance: order[(i + 1) % len(order)].instance
                for i in range(len(order))
            }
            normalized = [
                LayerAssignment(
                    instance=a.instance,
                    layers=a.layers,
                    next_instance=a.next_instance or ring_map.get(a.instance),
                    window_size=a.window_size,
                    residency_size=a.window_size,
                )
                for a in normalized
            ]

        # Persist manual topology
        self.topology = TopologyInfo(
            model=req.model,
            num_layers=num_layers,
            kv_bits=req.kv_bits,
            devices=devices_props,
            assignments=normalized,
        )
        logger.info(
            "Manual topology prepared: %d devices, %d layers",
            len(devices_props),
            int(num_layers),
        )
        return self.topology

    async def _handle_load_model(
        self, req: APILoadModelRequest
    ) -> APILoadModelResponse:
        """Handle load model request.

        Args:
            req: Load model request

        Returns:
            Load model response
        """
        # Decide model and assignments
        if self.topology:
            topology = self.topology

            # Always use kv_bits from topology; log if caller asked for different
            kv_bits_use = topology.kv_bits
            if req.kv_bits != kv_bits_use:
                logger.info(
                    "load_model request kv_bits %s overridden by topology kv_bits %s",
                    req.kv_bits,
                    kv_bits_use,
                )

            if topology.model:
                logger.info(
                    "load_model request model %s overridden by topology model %s",
                    req.model,
                    topology.model,
                )
                # use existing model from topology
                model_to_load = topology.model
            elif req.model:
                model_to_load = req.model
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "No model specified in request and no model in existing topology. "
                        "Call /v1/prepare_topology or include 'model' to bootstrap with discovery."
                    ),
                )
        elif req.model:
            # prepare topology on-the-fly
            topology = await self._handle_prepare_topology(
                PrepareTopologyRequest(
                    model=req.model, kv_bits=req.kv_bits, seq_len=req.seq_len
                )
            )
            model_to_load = req.model
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "No topology configured and no model provided. "
                    "Call /v1/prepare_topology or /v1/prepare_topology_manual first, "
                    "or include 'model' to bootstrap with discovery."
                ),
            )
            kv_bits_use = topology.kv_bits

        assignments_to_use = topology.assignments
        shards = {dev.instance: dev for dev in topology.devices}
        logger.info("Loading model: %s", model_to_load)

        # Notify each shard to load their layers via HTTP
        api_properties = await self.discovery.async_get_own_properties()
        shard_statuses: List[ShardLoadStatus] = []
        async with httpx.AsyncClient() as http_client:
            for assignment in assignments_to_use:
                instance = assignment.instance
                # Flatten layers for shard loading
                layers = [
                    layer
                    for round_layers in assignment.layers
                    for layer in round_layers
                ]

                if instance not in shards:
                    logger.warning("Shard %s not found in discovery", instance)
                    shard_statuses.append(
                        ShardLoadStatus(
                            instance=instance,
                            success=False,
                            message="Shard not found in discovery",
                            layers_loaded=[],
                        )
                    )
                    continue

                shard_props = shards[instance]

                # Get next node address from next_service in ring (if provided)
                next_shard = None
                if assignment.next_instance is not None:
                    ns = assignment.next_instance
                    if ns in shards:
                        next_shard = shards[ns]
                        logger.info("Shard %s next node in ring: %s", instance, ns)
                    else:
                        logger.info(
                            "Shard %s next instance %s not found; skipping ring hop",
                            instance,
                            ns,
                        )

                try:
                    # Build API callback address (gRPC)
                    api_callback_address = f"{api_properties.local_ip}:{self.grpc_port}"

                    # Call load_model via HTTP (window_size unified)
                    url = f"http://{shard_props.local_ip}:{shard_props.server_port}/load_model"
                    # Let the shard decide warmup based on its computed mode (fit/offload)
                    payload = ShardLoadModelRequest(
                        model_path=model_to_load,
                        layers=layers,
                        warmup=True,
                        next_node=next_shard,
                        window_size=assignment.window_size,
                        residency_size=assignment.residency_size,
                        total_layers=topology.num_layers,
                        kv_bits=kv_bits_use,
                        api_callback_address=api_callback_address,
                    ).model_dump()

                    # timeout is `None` because shards may actually be downloading weights for the first time
                    # TODO: can we do this in a better way?
                    response = await http_client.post(url, json=payload, timeout=None)
                    result = ShardLoadModelResponse.model_validate_json(response.text)

                    shard_statuses.append(
                        ShardLoadStatus(
                            instance=instance,
                            success=result.success,
                            message=result.message,
                            layers_loaded=result.layers_loaded,
                        )
                    )
                    logger.info(
                        "Shard %s load result: success=%s (%s)",
                        instance,
                        result.success,
                        result.message,
                    )
                except Exception as e:
                    logger.exception("Error loading model on shard %s: %s", instance, e)
                    shard_statuses.append(
                        ShardLoadStatus(
                            instance=instance,
                            success=False,
                            message=str(e),
                            layers_loaded=[],
                        )
                    )

        # Check if all shards loaded successfully
        if all(status.success for status in shard_statuses):
            try:
                # Load tokenizer without forcing weight downloads on API node
                tok_dir = resolve_tokenizer_dir(model_to_load)
                self.tokenizer = load_tokenizer(tok_dir, {})

                # Connect to first shard (head device)
                # this should be the first device in `self.topology.devices`
                await self._connect_first_shard()

                logger.info("API-side model loaded successfully for %s", model_to_load)
                return APILoadModelResponse(
                    model=model_to_load,
                    success=True,
                    shard_statuses=shard_statuses,
                )
            except Exception as e:
                logger.exception("Error loading API-side model: %s", e)
                return APILoadModelResponse(
                    model=model_to_load,
                    success=False,
                    shard_statuses=shard_statuses,
                    message=f"Error loading API-side model: {e}",
                )
        else:
            failed_shards = [
                status.instance for status in shard_statuses if not status.success
            ]
            logger.error("Failed to load model on shards: %s", failed_shards)
            return APILoadModelResponse(
                model=model_to_load,
                success=False,
                shard_statuses=shard_statuses,
            )

    async def _handle_unload_model(self) -> UnloadModelResponse:
        """Handle unload model request by unloading from all shards.

        Returns:
            Unload model response
        """
        logger.info("Unloading model from all shards")

        if not self.topology:
            logger.warning("No topology configured, nothing to unload")
            return UnloadModelResponse(
                success=True,
                shard_statuses=[],
                message="No topology configured, nothing to unload",
            )

        # Call unload_model on each shard via HTTP
        shard_statuses: List[ShardUnloadStatus] = []
        async with httpx.AsyncClient() as http_client:
            for shard in self.topology.devices:
                try:
                    # Call unload_model via HTTP
                    url = f"http://{shard.local_ip}:{shard.server_port}/unload_model"
                    response = await http_client.post(url, timeout=30.0)
                    result = response.json()  # FIXME: add shard response type

                    shard_statuses.append(
                        ShardUnloadStatus(
                            instance=shard.instance,
                            success=result.get("success", False),
                            message=result.get("message", ""),
                        )
                    )

                    logger.info(
                        "Shard %s unload result: success=%s, message=%s",
                        shard.instance,
                        result.get("success"),
                        result.get("message"),
                    )

                except Exception as e:
                    logger.exception(
                        "Error unloading model on shard %s: %s",
                        shard.instance,
                        e,
                    )
                    shard_statuses.append(
                        ShardUnloadStatus(
                            instance=shard.instance,
                            success=False,
                            message=f"Error: {str(e)}",
                        )
                    )

        # Check if all shards unloaded successfully
        all_success = all(status.success for status in shard_statuses)
        if not all_success:
            failed_shards = [
                status.instance for status in shard_statuses if not status.success
            ]
            logger.error(f"Failed to unload model on shards: {failed_shards}")

        # Unload API-side model components
        try:
            self.tokenizer = None
            self.model_metadata = None
            self.generate_step = None

            self.topology.model = None

            # Close first shard connection
            if self.first_shard_channel:
                await self.first_shard_channel.close()
                self.first_shard_channel = None
                self.first_shard_stub = None
                self.first_shard_address = None

            logger.info("API-side model unloaded successfully")

        except Exception as e:
            logger.exception(f"Error unloading API-side model: {e}")
            all_success = False
            shard_statuses.append(
                ShardUnloadStatus(
                    instance="api",
                    success=False,
                    message=f"API model unload error: {str(e)}",
                )
            )

        return UnloadModelResponse(
            success=all_success,
            shard_statuses=shard_statuses,
            message="Model unloaded successfully"
            if all_success
            else "Some shards failed to unload",
        )

    async def _connect_first_shard(self) -> bool:
        """Connect to the shard that owns layer 0 (entry shard).

        Falls back to the first device in topology when ownership cannot be
        determined. Returns True on success, False otherwise.
        """
        if not self.topology or not self.topology.devices:
            logger.error("No topology configured; cannot connect to first shard")
            return False

        # Pick the device whose assignment contains layer 0; fallback to index 0
        start_instance: str | None = None
        try:
            for assignment in self.topology.assignments:
                # Flatten round layers
                flat = [
                    layer
                    for round_layers in assignment.layers
                    for layer in round_layers
                ]
                if 0 in flat:
                    start_instance = assignment.instance
                    break
        except Exception:
            start_instance = None

        # find the start device w.r.t name
        start_device = None
        if start_instance is not None:
            try:
                for dev in self.topology.devices:
                    if dev.instance == start_instance:
                        start_device = dev
                        break
            except Exception:
                start_device = None

        if start_device is None:
            start_device = self.topology.devices[0]

        first_shard_address = f"{start_device.local_ip}:{start_device.shard_port}"
        if self.first_shard_channel:
            return True

        try:
            self.first_shard_channel = aio_grpc.insecure_channel(first_shard_address)
            self.first_shard_stub = DnetRingServiceStub(self.first_shard_channel)

            # Prepare generate_step with gRPC callback
            api_properties = await self.discovery.async_get_own_properties()
            # Prefer Thunderbolt IP for token callbacks if available
            tb_ip = None
            try:
                tb = api_properties.thunderbolt
                if tb and tb.ip_addr:
                    tb_ip = tb.ip_addr
            except Exception:
                tb_ip = None
            cb_ip = tb_ip or api_properties.local_ip
            callback_addr = f"{cb_ip}:{self.grpc_port}"
            self.generate_step = create_generate_step_for_ring_with_grpc(
                self.first_shard_stub,
                callback_protocol="grpc",
                callback_addr=callback_addr,
                compression=self._compression_pct,
            )

            logger.info("Connected to first shard at %s", first_shard_address)
            return True

        except Exception as e:
            logger.warning(
                "Failed to connect to first shard %s: %s", first_shard_address, e
            )
            self.first_shard_channel = None
            self.first_shard_stub = None
            self.generate_step = None
            return False

    async def _get_shards_from_discovery(self) -> Dict[str, DnetDeviceProperties]:
        """Get shards from discovery keyed by instance name (excluding managers)."""
        devices = await self.discovery.async_get_properties()
        # Normalize keys to the short "instance" form
        normalized: Dict[str, DnetDeviceProperties] = {}
        for _instance, props in devices.items():
            if props.is_manager:
                continue
            normalized[props.instance] = props
        return normalized

    async def _profile_model(
        self, repo_id: str, batch_sizes: List[int], sequence_length: int
    ) -> ModelProfile:
        """Profile model.

        Args:
            repo_id: Hugging Face repository ID
            batch_sizes: List of batch sizes to profile
            sequence_length: Sequence length to profile

        Returns:
            Model profile
        """
        model_profile_split = profile_model(
            repo_id=repo_id,
            batch_sizes=batch_sizes,
            sequence_length=sequence_length,
        )
        logger.info("Model profiling completed for %s.", repo_id)
        return model_profile_split.to_model_profile()

    async def _collect_shard_profiles(
        self,
        shards: Dict[str, DnetDeviceProperties],
        repo_id: str,
        embedding_size: int,
        max_batch_exp: int,
        batch_sizes: List[int],
        thunderbolt_conns: dict[str, dict[str, ThunderboltConnection]] = {},
    ) -> Dict[str, DeviceProfile]:
        """Collect profile data from all shards.

        Args:
            shards: Discovered shards
            repo_id: Model repository ID
            embedding_size: Model embedding size
            max_batch_exp: Maximum batch size exponent
            batch_sizes: List of batch sizes to profile
            thunderbolt_conns: Pre-discovered thunderbolt connections per shard

        Returns:
            Tuple of (collected shard profiles, thunderbolt connections)
        """
        # Calculate payload sizes
        base_size = embedding_size * 4  # 4*e due to paper
        payload_sizes = [base_size * batch_size for batch_size in batch_sizes]

        logger.info(
            "Model %s: embedding_size=%d, payload_sizes=%s",
            repo_id,
            embedding_size,
            payload_sizes,
        )

        async with httpx.AsyncClient() as client:
            # health-check all shards in parallel
            logger.info("Starting health checks for all shards...")
            health_tasks: list[asyncio._CoroutineLike[httpx.Response]] = []
            shard_list: list[tuple[str, DnetDeviceProperties]] = []
            for shard_name, shard_props in shards.items():
                if shard_props.is_manager:
                    logger.warning(
                        "Skipping manager node %s in profile collection", shard_name
                    )
                    continue

                shard_list.append((shard_name, shard_props))
                health_tasks.append(
                    client.get(
                        f"http://{shard_props.local_ip}:{shard_props.server_port}/health",
                        timeout=5.0,
                    )
                )

            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)

            # filter healthy shards
            healthy_shards: list[tuple[str, DnetDeviceProperties]] = []
            for (shard_name, shard_props), health_result in zip(
                shard_list, health_results
            ):
                if isinstance(health_result, Exception):
                    logger.warning(
                        "Health check failed for %s: %s", shard_name, health_result
                    )
                    continue
                elif isinstance(health_result, httpx.Response):
                    if health_result.status_code == 200:
                        healthy_shards.append((shard_name, shard_props))
                        logger.info("Health check passed for %s", shard_name)
                    else:
                        logger.warning(
                            "Health check failed for %s: status %s",
                            shard_name,
                            health_result.status_code,
                        )
                else:
                    pass

            logger.info("Healthy shards: %d/%d", len(healthy_shards), len(shard_list))
            if not healthy_shards:
                logger.error("No healthy shards found!")
                return {}

            # measure latencies on all healthy shards in parallel)
            logger.info("Measuring latencies for all healthy shards...")
            latency_tasks: list[asyncio._CoroutineLike[httpx.Response]] = []
            for shard_name, shard_props in healthy_shards:
                server_port, server_ip = shard_props.server_port, shard_props.local_ip
                latency_url = f"http://{server_ip}:{server_port}/measure_latency"
                latency_request = MeasureLatencyRequest(
                    devices=shards,
                    thunderbolts=thunderbolt_conns.get(shard_name, {}),
                    payload_sizes=payload_sizes,
                )
                latency_tasks.append(
                    client.post(
                        latency_url, json=latency_request.model_dump(), timeout=1000.0
                    )
                )
            latency_results = await asyncio.gather(
                *latency_tasks, return_exceptions=True
            )

            # store latency data for each shard
            shard_latencies: dict[str, LatencyResults] = {}
            final_healthy_shards = []
            for (shard_name, shard_props), latency_result in zip(
                healthy_shards, latency_results
            ):
                if isinstance(latency_result, Exception):
                    logger.warning(
                        "Latency measurement failed for %s: %s",
                        shard_name,
                        latency_result,
                    )
                    continue
                elif isinstance(latency_result, httpx.Response):
                    if latency_result.status_code == 200:
                        latency_data = MeasureLatencyResponse.model_validate(
                            latency_result.json()
                        )
                        shard_latencies[shard_name] = latency_data.latency
                        final_healthy_shards.append((shard_name, shard_props))
                        logger.info("Latency measurement succeeded for %s", shard_name)
                    else:
                        logger.warning(
                            "Latency measurement failed for %s: status %s",
                            shard_name,
                            latency_result.status_code,
                        )
                else:
                    pass  # unexpected case

            logger.info("Latencies collected from %d shards", len(shard_latencies))

            if not final_healthy_shards:
                logger.error("No shards with successful latency measurements!")
                return {}

            # group healthy shards by local_ip (same device), so that we can profile per-device
            shards_by_device: Dict[str, List[Tuple[str, DnetDeviceProperties]]] = {}
            for shard_name, shard_props in final_healthy_shards:
                local_ip = shard_props.local_ip
                if local_ip not in shards_by_device:
                    shards_by_device[local_ip] = []
                shards_by_device[local_ip].append((shard_name, shard_props))
            logger.info(
                "Grouped %d shards into %d devices",
                len(final_healthy_shards),
                len(shards_by_device),
            )

            # profile devices (parallel per device, sequential per shard within device)
            async def profile_device_shards(
                device_shards: List[Tuple[str, DnetDeviceProperties]],
            ) -> List[Tuple[str, DeviceProfile]]:
                profiles: List[Tuple[str, DeviceProfile]] = []

                for shard_name, shard_props in device_shards:
                    try:
                        profile_url = f"http://{shard_props.local_ip}:{shard_props.server_port}/profile"

                        logger.info(
                            "Calling /profile endpoint for shard %s at %s",
                            shard_name,
                            profile_url,
                        )

                        response = await client.post(
                            profile_url,
                            json=ShardProfileRequest(
                                repo_id=repo_id,
                                thunderbolts=thunderbolt_conns.get(shard_name, {}),
                                payload_sizes=payload_sizes,
                                max_batch_exp=max_batch_exp,
                                devices=shards,
                            ).model_dump(),
                            timeout=1000.0,
                        )

                        if response.status_code == 200:
                            profile_response = ShardProfileResponse.model_validate(
                                response.json()
                            )
                            profiles.append((shard_name, profile_response.profile))
                            logger.info(
                                "Successfully collected profile from %s", shard_name
                            )
                        else:
                            logger.error(
                                "Failed to get profile from %s: %s",
                                shard_name,
                                response.status_code,
                            )

                    except Exception as e:
                        logger.exception(
                            "Error calling /profile for %s: %s", shard_name, e
                        )

                return profiles

            # run profiling for all devices in parallel
            device_tasks = [
                profile_device_shards(device_shards)
                for device_shards in shards_by_device.values()
            ]
            device_results = await asyncio.gather(*device_tasks, return_exceptions=True)

            # merge latency data into device profiles
            shard_profiles: Dict[str, DeviceProfile] = {}
            for device_result in device_results:
                if isinstance(device_result, Exception):
                    logger.error("Device profiling failed: %s", device_result)
                    continue
                elif isinstance(device_result, list):
                    for shard_name, profile in device_result:
                        # set t_comm using median latency
                        if shard_name in shard_latencies:
                            median_latency = calculate_median_latency_seconds(
                                shard_latencies[shard_name]
                            )
                            if median_latency is not None:
                                profile.t_comm = float(median_latency)
                                logger.info(
                                    f"Set t_comm for {shard_name} to median latency: {profile.t_comm:.6f}s"
                                )
                            else:
                                logger.warning(
                                    f"No valid latency measurements for {shard_name}, keeping default t_comm"
                                )

                        shard_profiles[shard_name] = profile

        logger.info("Collected profiles from %d shards", len(shard_profiles))
        return shard_profiles

    # FIXME: move this to elsewhere
    async def _run_solver(
        self,
        shard_profiles: Dict[str, DeviceProfile],
        model_profile: ModelProfile,
        device_order: List[str],
        kv_bits: Literal["4bit", "8bit", "fp16"],
    ) -> HALDAResult:
        """Run distilp with model and device profiles.

        Args:
            shard_profiles: Collected shard profiles
            model_profile: Model profile
            device_order: Optimized device ordering (head first, TB-connected adjacent)

        Returns:
            Tuple of (device names in solver order, solver result)
        """

        sorted_shard_profiles = [
            shard_profiles[name] for name in device_order if name in shard_profiles
        ]
        if not sorted_shard_profiles:
            raise ValueError("No valid shard profiles found")

        # mark the first device as head, others as non-head
        for i, profile in enumerate(sorted_shard_profiles):
            profile.is_head = i == 0

        logger.info("Running solver with %d shard profiles", len(sorted_shard_profiles))

        solution = halda_solve(
            devs=sorted_shard_profiles,
            model=model_profile,
            mip_gap=1e-4,
            plot=False,
            kv_bits=kv_bits,
        )

        logger.info(
            "Solver completed: k=%d, objective=%d", solution.k, solution.obj_value
        )

        return solution

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
                    "Using tokenizer chat template, prompt preview: %s...",
                    prompt[:50],
                )
                return prompt
            except Exception as e:
                logger.warning(
                    "Failed to apply chat template: %s, falling back to default", e
                )

        # Fallback to default role mapping
        buf = StringIO()
        for msg in messages:
            role_prefix = getattr(role_mapping, msg.role, "")
            buf.write(f"{role_prefix}{msg.content}{role_mapping.stop}")
        return buf.getvalue().rstrip() + role_mapping.assistant

    async def _handle_completion(
        self,
        req: ChatParams,
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
        profile_enabled = bool(req.profile)
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
                node_origin=f"localhost:{self.http_port}",
                prompt=prompt,
                pending_requests=self.pending_requests,
                params=req,
            ),  # type: ignore
            arange(req.max_tokens or 0),
        ):
            if profile_enabled and t_first_token is None:
                t_first_token = time.perf_counter()
            detokenizer.add_token(token)
            tokens.append(token)

            if (logprobs is not None) and (req.logprobs) and logprobs.size != 0:
                sorted_indices = mx.argpartition(-logprobs, kth=req.top_logprobs - 1)
                top_indices = sorted_indices[: (req.top_logprobs or 0)]
                top_logprobs_array = logprobs[top_indices]
                top_token_info = zip(
                    top_indices.tolist(),  # type: ignore # FIXME: !!!
                    top_logprobs_array.tolist(),  # type: ignore # FIXME: !!!
                )
                top_tokens.append(dict(top_token_info))

            if logprobs is not None and logprobs.size != 0:
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
            req.model,
            completion_reason,
            len(prompt),
            len(tokens),
            token_logprobs=token_logprobs,
            top_logprobs=top_tokens,
            tokens=tokens,
            metrics=metrics,
        )

    async def _handle_text_completion(self, req: CompletionRequestModel):
        prompt = mx.array(self.tokenizer.encode(req.prompt))  # type: ignore
        stop_id_sequences: List[List[int]] = [
            self.tokenizer.encode(stop_word, add_special_tokens=False)  # type: ignore
            for stop_word in (req.stop or [])
        ]
        chat_resp = await self._handle_completion(req, prompt, stop_id_sequences)
        text = chat_resp.choices[0].message.content
        return {
            "id": chat_resp.id,
            "object": "text_completion",
            "model": chat_resp.model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": chat_resp.choices[0].finish_reason,
                }
            ],
            "usage": chat_resp.usage,
        }

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
        model: str,
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
                    logprobs=ChatLogProbs(
                        token_logprobs=token_logprobs or [],
                        top_logprobs=top_logprobs or [],
                        tokens=tokens,
                    ),
                )
            ],
            # FIXME: bit too many `or 0`'s here?
            usage=ChatUsage(
                prompt_tokens=prompt_token_count or 0,
                completion_tokens=completion_token_count or 0,
                total_tokens=(prompt_token_count or 0) + (completion_token_count or 0),
            ),
            metrics=metrics,
            created=int(time.time()),
            model=model,
        )

    def _stream_chat(self, req: ChatRequestModel):
        created = int(time.time())
        nonce = f"chatcmpl-{uuid.uuid4()}"

        async def gen():
            stop_id_sequences: List[List[int]] = [
                self.tokenizer.encode(stop_word, add_special_tokens=False)  # type: ignore
                for stop_word in (req.stop or [])
            ]
            prompt_text = await self._convert_chat(req.messages)
            prompt = mx.array(self.tokenizer.encode(prompt_text))  # type: ignore
            prompt_tokens = int(len(prompt))  # 1D token count
            t_start = time.perf_counter()
            t_first_token: Optional[float] = None
            detok = self.tokenizer.detokenizer  # type: ignore
            detok.reset()
            # Initial role delta
            chunk = {
                "id": nonce,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            tokens: list[int] = []
            async for (token, _), _ in azip(
                self.generate_step(
                    nonce=nonce,
                    node_origin=f"localhost:{self.http_port}",  # FIXME: Not sure of this, grpc-http port mix
                    prompt=prompt,
                    pending_requests=self.pending_requests,
                    params=req,
                ),  # type: ignore
                arange(req.max_tokens or 0),
            ):
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                tokens.append(token)
                detok.add_token(token)
                try:
                    delta = detok.delta
                except Exception:
                    delta = self.tokenizer.decode([token])  # type: ignore
                chunk = {
                    "id": nonce,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                stop = await self._stopping_criteria(
                    tokens,
                    stop_id_sequences,
                    self.tokenizer.eos_token_id,  # type: ignore
                )  # type: ignore
                if stop.stop_met:
                    break

            # Build end-of-stream chunk with usage + metrics
            t_end = time.perf_counter()
            total_s = max(t_end - t_start, 1e-9)
            gen_s = max(t_end - (t_first_token or t_start), 1e-9)
            tokens_generated = len(tokens)
            done = {
                "id": nonce,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": tokens_generated,
                    "total_tokens": prompt_tokens + tokens_generated,
                },
            }
            if req.profile:
                metrics = {
                    "total_ms": round(total_s * 1000.0, 3),
                    "ttfb_ms": round(((t_first_token or t_end) - t_start) * 1000.0, 3),
                    "token_gen_ms": round(gen_s * 1000.0, 3),
                    "tokens_generated": tokens_generated,
                    "tps_overall": round(
                        (tokens_generated / total_s) if tokens_generated else 0.0, 4
                    ),
                    "tps_decoding": round(
                        (tokens_generated / gen_s) if tokens_generated else 0.0, 4
                    ),
                }
                done["metrics"] = metrics
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return gen()

    def _stream_completion(self, req: CompletionRequestModel):
        created = int(time.time())
        nonce = f"cmpl-{uuid.uuid4()}"

        async def gen():
            prompt = mx.array(self.tokenizer.encode(req.prompt))  # type: ignore
            prompt_tokens = int(len(prompt))
            t_start = time.perf_counter()
            t_first_token: Optional[float] = None
            detok = self.tokenizer.detokenizer  # type: ignore
            detok.reset()
            out_tokens = 0
            async for (token, _), _ in azip(
                self.generate_step(
                    nonce=nonce,
                    node_origin=f"localhost:{self.http_port}",  # FIXME: Not sure of this, grpc-http port mix
                    prompt=prompt,
                    pending_requests=self.pending_requests,
                    params=req,
                ),  # type: ignore
                arange(req.max_tokens or 0),
            ):
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                out_tokens += 1
                detok.add_token(token)
                try:
                    delta = detok.delta
                except Exception:
                    delta = self.tokenizer.decode([token])  # type: ignore
                chunk = {
                    "id": nonce,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # End-of-stream chunk with usage + metrics
            t_end = time.perf_counter()
            total_s = max(t_end - t_start, 1e-9)
            gen_s = max(t_end - (t_first_token or t_start), 1e-9)
            done = {
                "id": nonce,
                "object": "text_completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": out_tokens,
                    "total_tokens": prompt_tokens + out_tokens,
                },
            }
            if req.profile:
                metrics = {
                    "total_ms": round(total_s * 1000.0, 3),
                    "ttfb_ms": round(((t_first_token or t_end) - t_start) * 1000.0, 3),
                    "token_gen_ms": round(gen_s * 1000.0, 3),
                    "tokens_generated": out_tokens,
                    "tps_overall": round(
                        (out_tokens / total_s) if out_tokens else 0.0, 4
                    ),
                    "tps_decoding": round(
                        (out_tokens / gen_s) if out_tokens else 0.0, 4
                    ),
                }
                done["metrics"] = metrics
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return gen()

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

        # Close channels
        if self.first_shard_channel:
            await self.first_shard_channel.close()

        # Stop gRPC server
        if self.api_grpc_server:
            await self.api_grpc_server.stop(grace=5)

        # Stop discovery
        if self.discovery.is_running():
            logger.info("Stopping discovery service for API node")
            await self.discovery.async_stop()
            await self.discovery.async_free_instance()
        else:
            logger.warning("Discovery service was not running")

        logger.info("API server shutdown complete")
