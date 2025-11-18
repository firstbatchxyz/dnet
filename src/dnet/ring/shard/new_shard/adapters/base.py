"""
TopologyAdapter base: takes a ShardRuntime, wires transport in/out.
Defines “ingress” and “egress” hooks but no concrete protocol.
"""

from abc import ABC, abstractmethod
import asyncio
from dnet.protos.dnet_ring_pb2 import ActivationRequest
from dnet.ring.data_types import ActivationMessage
from dnet.ring.shard.models import ShardUnloadModelResponse
from dnet.utils.logger import logger
import mlx.core as mx

class TopologyAdapter(ABC):
    """
    Base class for topology adapters.
    """

    def __init__(self, runtime, discovery):
        self.runtime = runtime
        self.discovery = discovery
        self.running = False

    @property
    @abstractmethod
    def ingress_q(self) -> asyncio.Queue[ActivationRequest]:
        """Required queue property. Filled by incoming GRPC/network data."""
        pass

    @property
    @abstractmethod
    def activation_computed_queue(self) -> asyncio.Queue[ActivationMessage]:
        pass

    @property
    @abstractmethod
    def activation_token_queue(self) -> asyncio.Queue[ActivationMessage]:
        pass

    @abstractmethod
    async def start(self):
        """Start any necessary background tasks for the adapter."""
        pass

    @abstractmethod
    async def ingress(self, req: ActivationRequest):
        """Handle incoming data to the shard.
        Ingress = network -> adapter -> runtime.
        """
        pass

    @abstractmethod
    async def egress(self, msg: ActivationMessage):
        """Handle outgoing data from the shard.
        Egress = runtime -> adapter -> network.
        """
        pass

    @abstractmethod
    async def configure_topology(self, req):
        """Configure the adapter for a new model load."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the adapter and any background tasks."""
        pass
