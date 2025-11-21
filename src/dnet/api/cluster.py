import httpx
import asyncio
from typing import Dict, Optional, List, Any, Tuple, Literal
from dnet_p2p import (
    AsyncDnetP2P,
    DnetDeviceProperties,
    ThunderboltConnection,
    discover_all_thunderbolt_connections,
)
from distilp.common import DeviceProfile
from dnet.core.topology import TopologySolver
from dnet.core.types.topology import TopologyInfo
from dnet.utils.logger import logger
from dnet.utils.latency import LatencyResults, calculate_median_latency_seconds
from dnet.shard.models import (
    MeasureLatencyRequest,
    MeasureLatencyResponse,
    ShardProfileRequest,
    ShardProfileResponse,
)


class ClusterManager:
    def __init__(self, discovery: AsyncDnetP2P, solver: TopologySolver):
        self.discovery = discovery
        self.solver = solver
        self.current_topology: Optional[TopologyInfo] = None
        self.device_profiles: Dict[str, DeviceProfile] = {}
        self.shards: Dict[str, DnetDeviceProperties] = {}
        self.thunderbolts: Dict[str, Dict[str, ThunderboltConnection]] = {}

    async def scan_devices(self) -> Dict[str, DnetDeviceProperties]:
        """Query discovery for current peers and cache them."""
        peers = await self.discovery.async_get_properties()
        self.shards = peers
        return self.shards

    async def profile_cluster(
        self,
        model_id: str,
        embedding_size: int,
        max_batch_exp: int,
        batch_sizes: List[int],
    ) -> Dict[str, DeviceProfile]:
        """Orchestrates the /profile calls to devices and aggregates results."""
        shards = self.shards
        if not shards:
            logger.warning("No shards available for profiling")
            return {}

        # Discover thunderbolt connections
        self.thunderbolts = discover_all_thunderbolt_connections(shards)

        # Calculate payload sizes
        base_size = embedding_size * 4  # 4*e due to paper
        payload_sizes = [base_size * batch_size for batch_size in batch_sizes]

        logger.info(
            "Model %s: embedding_size=%d, payload_sizes=%s",
            model_id,
            embedding_size,
            payload_sizes,
        )

        async with httpx.AsyncClient() as client:
            # health-check all shards in parallel
            logger.info("Starting health checks for all shards...")
            health_tasks: list[asyncio.Task] = []
            shard_list: list[tuple[str, DnetDeviceProperties]] = []
            for shard_name, shard_props in shards.items():
                if shard_props.is_manager:
                    continue

                shard_list.append((shard_name, shard_props))
                health_tasks.append(
                    asyncio.create_task(
                        client.get(
                            f"http://{shard_props.local_ip}:{shard_props.server_port}/health",
                            timeout=5.0,
                        )
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
                    else:
                        logger.warning(
                            "Health check failed for %s: status %s",
                            shard_name,
                            health_result.status_code,
                        )

            logger.info("Healthy shards: %d/%d", len(healthy_shards), len(shard_list))
            if not healthy_shards:
                logger.error("No healthy shards found!")
                return {}

            # measure latencies on all healthy shards in parallel
            logger.info("Measuring latencies for all healthy shards...")
            latency_tasks: list[asyncio.Task] = []
            for shard_name, shard_props in healthy_shards:
                server_port, server_ip = shard_props.server_port, shard_props.local_ip
                latency_url = f"http://{server_ip}:{server_port}/measure_latency"
                latency_request = MeasureLatencyRequest(
                    devices=shards,
                    thunderbolts=self.thunderbolts.get(shard_name, {}),
                    payload_sizes=payload_sizes,
                )
                latency_tasks.append(
                    asyncio.create_task(
                        client.post(
                            latency_url,
                            json=latency_request.model_dump(),
                            timeout=1000.0,
                        )
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
                    else:
                        logger.warning(
                            "Latency measurement failed for %s: status %s",
                            shard_name,
                            latency_result.status_code,
                        )

            if not final_healthy_shards:
                logger.error("No shards with successful latency measurements!")
                return {}

            # group healthy shards by local_ip (same device)
            shards_by_device: Dict[str, List[Tuple[str, DnetDeviceProperties]]] = {}
            for shard_name, shard_props in final_healthy_shards:
                local_ip = shard_props.local_ip
                if local_ip not in shards_by_device:
                    shards_by_device[local_ip] = []
                shards_by_device[local_ip].append((shard_name, shard_props))

            # profile devices
            async def profile_device_shards(
                device_shards: List[Tuple[str, DnetDeviceProperties]],
            ) -> List[Tuple[str, DeviceProfile]]:
                profiles: List[Tuple[str, DeviceProfile]] = []

                for shard_name, shard_props in device_shards:
                    try:
                        profile_url = f"http://{shard_props.local_ip}:{shard_props.server_port}/profile"
                        response = await client.post(
                            profile_url,
                            json=ShardProfileRequest(
                                repo_id=model_id,
                                thunderbolts=self.thunderbolts.get(shard_name, {}),
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
            self.device_profiles = {}
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
                            else:
                                logger.warning(
                                    f"No valid latency measurements for {shard_name}, keeping default t_comm"
                                )

                        self.device_profiles[shard_name] = profile

        logger.info("Collected profiles from %d shards", len(self.device_profiles))
        return self.device_profiles

    async def solve_topology(
        self,
        profiles: Dict[str, DeviceProfile],
        model_profile: Any,  # ModelProfile
        model_name: str,
        num_layers: int,
        kv_bits: Literal["4bit", "8bit", "fp16"],
    ) -> TopologyInfo:
        """Delegates to the configured solver."""

        self.current_topology = await self.solver.solve(
            profiles=profiles,
            model_profile=model_profile,
            model_name=model_name,
            num_layers=num_layers,
            kv_bits=kv_bits,
            shards=self.shards,
            thunderbolts=self.thunderbolts,
        )
        return self.current_topology

    def get_head_node(self) -> Optional[DnetDeviceProperties]:
        """Returns the device responsible for Layer 0."""
        if not self.current_topology:
            return None
        # Logic to find layer 0 holder
        for assignment in self.current_topology.assignments:
            for layer_range in assignment.layers:
                if 0 in layer_range:
                    return self.shards.get(assignment.instance)
        return None
