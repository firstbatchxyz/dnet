"""
TopologyAdapter base: takes a ShardRuntime, wires transport in/out.
Defines “ingress” and “egress” hooks but no concrete protocol.
"""
from abc import ABC,abstractmethod
import asyncio
from .....protos.dnet_ring_pb2 import ActivationRequest
from ....data_types import ActivationMessage


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
    async def ingress(self, data):
        """Handle incoming data to the shard.
        Ingress = network -> adapter -> runtime.
        """
        pass

    @abstractmethod
    async def egress(self, data):
        """Handle outgoing data from the shard.
        Egress = runtime -> adapter -> network.
        """
        pass

    @abstractmethod
    async def configure_for_model(self, req):
        """Configure the adapter for a new model load."""
        pass