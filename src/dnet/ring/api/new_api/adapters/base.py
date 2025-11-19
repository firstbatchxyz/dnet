from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional


class ApiAdapterBase(ABC):
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
    async def send_tokens(self, nonce: str, tokens: bytes, callback_addr: str) -> None: ...

    @abstractmethod
    async def await_token(self, nonce: str, timeout_s: float) -> int: ...

    @abstractmethod
    def resolve_token(self, nonce: str, token_id: int) -> None: ...

