from abc import ABC, abstractmethod
from typing import Any
from dnet.core.types.messages import TokenResult
from dnet.core.topology import TopologySolver


class ApiAdapterBase(ABC):
    """Abstract base class for API-Shard communication adapters."""

    def __init__(self) -> None:
        self.running = False

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    async def connect_first_shard(self, ip: str, port: int) -> None: ...

    @abstractmethod
    async def reset_cache(self) -> None: ...

    @abstractmethod
    async def send_tokens(
        self,
        nonce: str,
        tokens: bytes,
        callback_addr: str,
        logprobs: bool = False,
        top_logprobs: int = 0,
        decoding_config: Any = None,  # DecodingConfig
        start_pos: int = 0,
    ) -> None: ...

    @abstractmethod
    async def await_token(self, nonce: str, timeout_s: float) -> TokenResult: ...

    @abstractmethod
    def resolve_token(self, nonce: str, result: TokenResult) -> None: ...


class Strategy(ABC):
    """
    Bundles a TopologySolver and an ApiAdapterBase for a specific execution strategy.
    """

    @property
    @abstractmethod
    def solver(self) -> TopologySolver: ...

    @property
    @abstractmethod
    def adapter(self) -> ApiAdapterBase: ...
