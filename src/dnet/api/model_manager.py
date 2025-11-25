import time

import httpx
from typing import Optional, Dict, List, Any, Callable
from mlx_lm.tokenizer_utils import load_tokenizer

from dnet_p2p import DnetDeviceProperties
from dnet.utils.logger import logger
from dnet.utils.model import resolve_tokenizer_dir
from dnet.core.types.topology import TopologyInfo
from .catalog import model_catalog
from .models import (
    APILoadModelResponse,
    ShardLoadStatus,
    UnloadModelResponse,
    ShardUnloadStatus,
    ModelObjectExtended,
)
from dnet.shard.models import ShardLoadModelRequest, ShardLoadModelResponse


class ModelManager:
    def __init__(
        self,
        on_model_change: Optional[Callable[[Optional[str], int, bool], None]] = None,
    ) -> None:
        self.current_model_id: Optional[str] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.tokenizer: Optional[Any] = None
        self.on_model_change = on_model_change

        self.available_models: List[ModelObjectExtended] = []
        for model in model_catalog["models"]:
            extended_model = ModelObjectExtended(
                id=model["id"],
                arch=model["arch"],
                quantization=model["quantization"],
                alias=model["alias"],
                created=int(model["created"])
                if "created" in model
                else int(time.time()),
            )
            self.available_models.append(extended_model)

    def is_model_available(self, model_id: str) -> bool:
        """
        Checks if the given model ID is available in the catalog.
        """
        for model in self.available_models:
            if model.id == model_id:
                return True
        return False

    async def load_model(
        self,
        topology: TopologyInfo,
        api_properties: DnetDeviceProperties,
        grpc_port: int,
    ) -> APILoadModelResponse:
        """
        Orchestrates model loading across the cluster based on topology.
        """
        if topology.model is None:
            raise ValueError("Topology must specify a model to load")

        model_to_load: str = topology.model
        logger.info(
            f"Loading model {model_to_load} across {len(topology.assignments)} shards"
        )

        assignments_to_use = topology.assignments
        shards = {dev.instance: dev for dev in topology.devices}

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
                    else:
                        logger.info(
                            "Shard %s next instance %s not found; skipping ring hop",
                            instance,
                            ns,
                        )

                try:
                    # Build API callback address (gRPC)
                    api_callback_address = f"{api_properties.local_ip}:{grpc_port}"

                    # Call load_model via HTTP (window_size unified)
                    url = f"http://{shard_props.local_ip}:{shard_props.server_port}/load_model"

                    payload = ShardLoadModelRequest(
                        model_path=model_to_load,
                        layers=layers,
                        warmup=True,
                        next_node=next_shard,
                        window_size=assignment.window_size,
                        residency_size=assignment.residency_size,
                        total_layers=topology.num_layers,
                        kv_bits=topology.kv_bits,
                        api_callback_address=api_callback_address,
                    ).model_dump()

                    # timeout is `None` because shards may actually be downloading weights
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
                # Load tokenizer
                tok_dir = resolve_tokenizer_dir(model_to_load)
                self.tokenizer = load_tokenizer(tok_dir, {})
                self.current_model_id = model_to_load

                logger.info("API-side model loaded successfully for %s", model_to_load)
                if self.on_model_change:
                    self.on_model_change(model_to_load, topology.num_layers, True)
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
            return APILoadModelResponse(
                model=model_to_load,
                success=False,
                shard_statuses=shard_statuses,
                message="One or more shards failed to load model",
            )

    async def unload_model(
        self, shards: Dict[str, DnetDeviceProperties]
    ) -> UnloadModelResponse:
        """
        Unloads model from all shards.
        """
        if not self.current_model_id:
            return UnloadModelResponse(
                success=True, message="No model loaded", shard_statuses=[]
            )

        logger.info(f"Unloading model {self.current_model_id}")

        shard_statuses: List[ShardUnloadStatus] = []

        async with httpx.AsyncClient() as client:
            for shard_name, shard_props in shards.items():
                if shard_props.is_manager:
                    continue

                try:
                    url = f"http://{shard_props.local_ip}:{shard_props.server_port}/unload_model"
                    response = await client.post(url, timeout=10.0)

                    if response.status_code == 200:
                        shard_statuses.append(
                            ShardUnloadStatus(
                                instance=shard_name, success=True, message="Unloaded"
                            )
                        )
                    else:
                        shard_statuses.append(
                            ShardUnloadStatus(
                                instance=shard_name,
                                success=False,
                                message=f"Status {response.status_code}",
                            )
                        )

                except Exception as e:
                    logger.error(f"Failed to unload model on {shard_name}: {e}")
                    shard_statuses.append(
                        ShardUnloadStatus(
                            instance=shard_name, success=False, message=str(e)
                        )
                    )

        self.current_model_id = None
        self.tokenizer = None
        if self.on_model_change:
            self.on_model_change(None, 0, False)

        return UnloadModelResponse(
            success=all(s.success for s in shard_statuses),
            message="Model unloaded",
            shard_statuses=shard_statuses,
        )
