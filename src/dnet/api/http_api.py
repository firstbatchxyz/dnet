from typing import Optional, Any, List
import asyncio
from hypercorn import Config
from hypercorn.utils import LifespanFailureError
import hypercorn.asyncio as aio_hypercorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from dnet.utils.model import get_model_config_json
from distilp.profiler import profile_model
from dnet.utils.logger import logger
from .models import (
    ChatRequestModel,
    APILoadModelRequest,
    APILoadModelResponse,
    PrepareTopologyRequest,
    PrepareTopologyManualRequest,
    UnloadModelResponse,
)
from dnet.core.types.topology import TopologyInfo, LayerAssignment
from dnet.shard.models import HealthResponse
from .cluster import ClusterManager
from .inference import InferenceManager
from .model_manager import ModelManager
from dnet_p2p import DnetDeviceProperties


class HTTPServer:
    def __init__(
        self,
        http_port: int,
        cluster_manager: ClusterManager,
        inference_manager: InferenceManager,
        model_manager: ModelManager,
        node_id: str,
    ):
        self.http_port = http_port
        self.cluster_manager = cluster_manager
        self.inference_manager = inference_manager
        self.model_manager = model_manager
        self.node_id = node_id
        self.app = FastAPI()
        self.http_server: Optional[asyncio.Task] = None

    async def start(self, shutdown_trigger: Any = lambda: asyncio.Future()) -> None:
        await self._setup_routes()

        config = Config.from_mapping(
            bind=f"0.0.0.0:{self.http_port}", log_level="info", use_reloader=False
        )

        self.http_server = asyncio.create_task(
            aio_hypercorn.serve(self.app, config, shutdown_trigger=shutdown_trigger)  # type: ignore
        )

    async def shutdown(self) -> None:
        if self.http_server and not self.http_server.done():
            self.http_server.cancel()
            try:
                await self.http_server
            except (asyncio.CancelledError, LifespanFailureError):
                pass

    async def wait_closed(self, timeout: float = 5.0) -> bool:
        if not self.http_server:
            return True
        try:
            await asyncio.wait_for(self.http_server, timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _setup_routes(self) -> None:
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route(
            "/v1/chat/completions", self.chat_completions, methods=["POST"]
        )
        self.app.add_api_route("/v1/load_model", self.load_model, methods=["POST"])
        self.app.add_api_route("/v1/unload_model", self.unload_model, methods=["POST"])
        # Topology endpoints
        self.app.add_api_route("/v1/topology", self.get_topology, methods=["GET"])
        self.app.add_api_route(
            "/v1/prepare_topology", self.prepare_topology, methods=["POST"]
        )
        self.app.add_api_route(
            "/v1/prepare_topology_manual",
            self.prepare_topology_manual,
            methods=["POST"],
        )
        self.app.add_api_route("/v1/devices", self.get_devices, methods=["GET"])

    async def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            node_id=0,  # TODO: Doublecheck, Deprecated field, use 'instance' instead
            running=True,
            model_loaded=self.model_manager.current_model_id is not None,
            model_path=self.model_manager.current_model_id,
            assigned_layers=[],
            queue_size=0,
            grpc_port=0,
            http_port=self.http_port,
            instance=self.node_id,
        )

    async def chat_completions(self, req: ChatRequestModel):
        if not self.model_manager.current_model_id:
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded. Please load a model via /v1/load_model first.",
            )

        if req.stream:

            async def stream_generator():
                async for chunk in self.inference_manager.generate_stream(req):
                    # Use model_dump_json with exclude_none to omit empty fields like 'message' in chunks
                    data = chunk.model_dump_json(exclude_none=True)
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            return await self.inference_manager.chat_completions(req)

    async def load_model(self, req: APILoadModelRequest) -> APILoadModelResponse:
        try:
            topology = self.cluster_manager.current_topology
            if topology is None:
                if not req.model:
                    return APILoadModelResponse(
                        model=req.model or "",
                        success=False,
                        shard_statuses=[],
                        message=(
                            "No topology configured. Call /v1/prepare_topology or "
                            "/v1/prepare_topology_manual first, or include 'model' to bootstrap."
                        ),
                    )

                model_config = get_model_config_json(req.model)
                embedding_size = int(model_config["hidden_size"])
                num_layers = int(model_config["num_hidden_layers"])

                await self.cluster_manager.scan_devices()
                batch_sizes = [1]
                profiles = await self.cluster_manager.profile_cluster(
                    req.model, embedding_size, 2, batch_sizes
                )
                if not profiles:
                    return APILoadModelResponse(
                        model=req.model,
                        success=False,
                        shard_statuses=[],
                        message="No profiles collected",
                    )

                model_profile_split = profile_model(
                    repo_id=req.model,
                    batch_sizes=batch_sizes,
                    sequence_length=req.seq_len,
                )
                model_profile = model_profile_split.to_model_profile()
                topology = await self.cluster_manager.solve_topology(
                    profiles, model_profile, req.model, num_layers, req.kv_bits
                )
                self.cluster_manager.current_topology = topology

            api_props = await self.cluster_manager.discovery.async_get_own_properties()
            response = await self.model_manager.load_model(
                topology, api_props, self.inference_manager.grpc_port
            )
            if response.success:
                first_shard = topology.devices[0]
                await self.inference_manager.connect_to_ring(
                    first_shard.local_ip, first_shard.shard_port, api_props.local_ip
                )
            return response

        except Exception as e:
            logger.exception("Error in load_model: %s", e)
            return APILoadModelResponse(
                model=req.model or "",
                success=False,
                shard_statuses=[],
                message=str(e),
            )

    async def unload_model(self) -> UnloadModelResponse:
        await self.cluster_manager.scan_devices()
        shards = self.cluster_manager.shards
        response = await self.model_manager.unload_model(shards)
        if response.success:
            self.cluster_manager.current_topology = None
        return response

    async def get_devices(self) -> JSONResponse:
        devices = await self.cluster_manager.discovery.async_get_properties()
        devices_dict = {
            instance: device_props.model_dump()
            for instance, device_props in devices.items()
        }
        return JSONResponse(content={"devices": devices_dict})

    async def get_topology(self) -> TopologyInfo:
        topo = self.cluster_manager.current_topology
        if topo is None:
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No topology configured. Call /v1/prepare_topology first.",
            )
        return topo

    async def prepare_topology(self, req: PrepareTopologyRequest) -> TopologyInfo:
        try:
            model_config = get_model_config_json(req.model)
            embedding_size = int(model_config["hidden_size"])
            num_layers = int(model_config["num_hidden_layers"])

            await self.cluster_manager.scan_devices()

            batch_sizes = [1]
            profiles = await self.cluster_manager.profile_cluster(
                req.model,
                embedding_size,
                req.max_batch_exp,
                batch_sizes,
            )
            if not profiles:
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No profiles collected from shards",
                )

            model_profile_split = profile_model(
                repo_id=req.model,
                batch_sizes=batch_sizes,
                sequence_length=req.seq_len,
            )
            model_profile = model_profile_split.to_model_profile()

            topology = await self.cluster_manager.solve_topology(
                profiles, model_profile, req.model, num_layers, req.kv_bits
            )
            self.cluster_manager.current_topology = topology
            return topology
        except Exception as e:
            logger.exception("Error in prepare_topology: %s", e)
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    async def prepare_topology_manual(
        self, req: PrepareTopologyManualRequest
    ) -> TopologyInfo:
        try:
            # Validate unique device names
            names = [d.instance for d in req.devices]
            if len(names) != len(set(names)):
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Device names must be unique in manual topology",
                )

            # Normalize assignments and ensure next_instance ring when missing
            name_set = set(names)
            norm: List[LayerAssignment] = []
            for a in req.assignments:
                if a.instance not in name_set:
                    from fastapi import HTTPException, status

                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Assignment references unknown device: {a.instance}",
                    )
                norm.append(
                    LayerAssignment(
                        instance=a.instance,
                        layers=a.layers,
                        next_instance=a.next_instance,
                        window_size=a.window_size,
                        residency_size=a.window_size,
                    )
                )

            if any(a.next_instance is None for a in norm) and len(norm) > 1:
                order = sorted(
                    norm,
                    key=lambda aa: min([layer for rr in aa.layers for layer in rr])
                    if aa.layers
                    else (1 << 30),
                )
                ring_map = {
                    order[i].instance: order[(i + 1) % len(order)].instance
                    for i in range(len(order))
                }
                norm = [
                    LayerAssignment(
                        instance=a.instance,
                        layers=a.layers,
                        next_instance=a.next_instance or ring_map.get(a.instance),
                        window_size=a.window_size,
                        residency_size=a.window_size,
                    )
                    for a in norm
                ]

            num_layers = req.num_layers
            if num_layers is None:
                flat = [layer for aa in norm for rr in aa.layers for layer in rr]
                if not flat:
                    from fastapi import HTTPException, status

                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No layers provided in assignments",
                    )
                num_layers = max(flat) + 1

            devices_props = [
                DnetDeviceProperties(
                    is_manager=False,
                    is_busy=False,
                    instance=d.instance,
                    server_port=d.server_port,
                    shard_port=d.shard_port,
                    local_ip=d.local_ip,
                )
                for d in req.devices
            ]

            topology = TopologyInfo(
                model=req.model,
                kv_bits=req.kv_bits,
                num_layers=int(num_layers),
                devices=devices_props,
                assignments=norm,
                solution=None,
            )
            self.cluster_manager.current_topology = topology
            return topology
        except Exception as e:
            logger.exception("Error in prepare_topology_manual: %s", e)
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
