import os
from dnet.utils.logger import logger
from dnet.utils.model import get_model_config_json
from distilp.profiler import profile_model
from dnet.core.types.topology import TopologyInfo
from .models import APILoadModelResponse, UnloadModelResponse


async def get_api_callback_address(
    cluster_manager,
    grpc_port: int | str,
) -> str:
    api_props = await cluster_manager.discovery.async_get_own_properties()
    grpc_port_int = int(grpc_port)
    api_callback_addr = (os.getenv("DNET_API_CALLBACK_ADDR") or "").strip()
    if not api_callback_addr:
        api_callback_addr = f"{api_props.local_ip}:{grpc_port_int}"
        if api_props.local_ip in ("127.0.0.1", "localhost"):
            logger.warning(
                "API callback address is loopback (%s). Remote shards will fail to SendToken. "
                "Set DNET_API_CALLBACK_ADDR to a reachable host:port.",
                api_callback_addr,
            )
    return api_callback_addr


async def _prepare_topology_core(
    cluster_manager,
    model: str,
    kv_bits: str,
    seq_len: int,
    progress_callback=None,
) -> TopologyInfo:
    model_config = get_model_config_json(model)
    embedding_size = int(model_config["hidden_size"])
    num_layers = int(model_config["num_hidden_layers"])

    await cluster_manager.scan_devices()
    if progress_callback:
        await progress_callback("Profiling cluster performance")
    batch_sizes = [1]
    profiles = await cluster_manager.profile_cluster(
        model, embedding_size, 2, batch_sizes
    )
    if not profiles:
        raise RuntimeError("No profiles collected")

    if progress_callback:
        await progress_callback("Computing optimal layer distribution")
    model_profile_split = profile_model(
        repo_id=model,
        batch_sizes=batch_sizes,
        sequence_length=seq_len,
    )
    model_profile = model_profile_split.to_model_profile()

    topology = await cluster_manager.solve_topology(
        profiles, model_profile, model, num_layers, kv_bits
    )
    return topology


async def _load_model_core(
    cluster_manager,
    model_manager,
    inference_manager,
    topology: TopologyInfo,
) -> APILoadModelResponse:
    api_props = await cluster_manager.discovery.async_get_own_properties()
    grpc_port = int(inference_manager.grpc_port)

    api_callback_addr = await get_api_callback_address(
        cluster_manager, inference_manager.grpc_port
    )
    response = await model_manager.load_model(
        topology,
        api_props,
        grpc_port,
        api_callback_address=api_callback_addr,
    )
    if response.success:
        first_shard = topology.devices[0]
        await inference_manager.connect_to_ring(
            first_shard.local_ip, first_shard.shard_port, api_callback_addr
        )
    return response


async def _unload_model_core(
    cluster_manager,
    model_manager,
) -> UnloadModelResponse:
    await cluster_manager.scan_devices()
    shards = cluster_manager.shards
    response = await model_manager.unload_model(shards)
    if response.success:
        cluster_manager.current_topology = None
    return response
