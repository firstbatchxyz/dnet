from abc import ABC, abstractmethod
from typing import Dict, Any
from dnet_p2p import DnetDeviceProperties, ThunderboltConnection
from distilp.common import DeviceProfile
from dnet.core.types.topology import TopologyInfo


class TopologySolver(ABC):
    """Abstract base class for topology solvers."""

    @abstractmethod
    async def solve(
        self,
        profiles: Dict[str, DeviceProfile],
        model_profile: Any,  # ModelProfile
        model_name: str,
        num_layers: int,
        kv_bits: str,
        shards: Dict[str, DnetDeviceProperties],
        thunderbolts: Dict[str, Dict[str, ThunderboltConnection]],
    ) -> TopologyInfo:
        """
        Computes the topology (layer assignments) for the given cluster and model.
        """
        pass
