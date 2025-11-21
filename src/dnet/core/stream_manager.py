"""Generic async stream lifecycle management.

Provides a reusable manager for request-scoped streaming RPCs.
Does not depend on ring-specific prototypes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from dnet.utils.logger import logger


@dataclass
class StreamContext:
    nonce: str
    queue: asyncio.Queue
    call: Optional[Any] = None
    ack_task: Optional[asyncio.Task] = None
    open: bool = False
    disabled: bool = False
    disabled_until: float = 0.0
    last_seq: int = 0
    last_activity_t: float = 0.0


CallFactory = Callable[[Any], Any]


class StreamManager:
    """Reusable stream lifecycle management.

    Owns per-nonce stream contexts, ACK reader, and idle cleanup.
    The owner provides a `call_factory` that accepts an async iterator and
    returns an async response stream (e.g., gRPC bidirectional stream).
    """

    def __init__(self, *, idle_timeout_s: float = 30.0, backoff_s: float = 0.5) -> None:
        self._streams: Dict[str, StreamContext] = {}
        self._idle_timeout_s = float(idle_timeout_s)
        self._backoff_s = float(backoff_s)

    def get_ctx(self, nonce: str) -> Optional[StreamContext]:
        return self._streams.get(nonce)

    async def get_or_create_stream(
        self, nonce: str, call_factory: CallFactory
    ) -> Optional[StreamContext]:
        ctx = self._streams.get(nonce)
        if ctx and ctx.open:
            try:
                loop = asyncio.get_running_loop()
                if ctx.disabled and loop.time() >= ctx.disabled_until:
                    ctx.disabled = False
            except Exception:
                pass
            return ctx

        ctx = StreamContext(nonce=nonce, queue=asyncio.Queue(maxsize=64))
        self._streams[nonce] = ctx

        async def _req_iter():
            while True:
                item = await ctx.queue.get()
                if item is None:
                    break
                yield item

        call = call_factory(_req_iter())
        ctx.call = call
        ctx.open = True
        ctx.last_activity_t = asyncio.get_running_loop().time()

        async def _ack_reader():
            try:
                async for ack in call:
                    # Treat explicit negative-acks or backpressure messages as temporary disables
                    if not getattr(ack, "accepted", True):
                        logger.debug(
                            "[STREAM][ACK] nonce=%s seq=%s accepted=0 msg=%s",
                            getattr(ack, "nonce", ""),
                            getattr(ack, "seq", -1),
                            getattr(ack, "message", ""),
                        )
                    msg = str(getattr(ack, "message", "")).lower()
                    if "backpressure" in msg:
                        loop = asyncio.get_running_loop()
                        ctx.disabled = True
                        ctx.disabled_until = loop.time() + self._backoff_s
            except Exception as e:
                logger.error("[STREAM] ack reader error: %s", e)
                ctx.open = False
                ctx.disabled = True

        ctx.ack_task = asyncio.create_task(_ack_reader())
        return ctx

    async def end_stream(self, nonce: str) -> None:
        ctx = self._streams.pop(nonce, None)
        if not ctx:
            return
        try:
            if ctx.ack_task is not None:
                ctx.ack_task.cancel()
        except Exception:
            pass
        try:
            # Close request iterator and underlying call
            await ctx.queue.put(None)
            if ctx.open and ctx.call is not None and hasattr(ctx.call, "aclose"):
                await ctx.call.aclose()
        except Exception:
            pass

    async def cleanup_idle_streams(self) -> int:
        now = asyncio.get_running_loop().time()
        closed = 0
        try:
            for nonce, ctx in list(self._streams.items()):
                if (now - ctx.last_activity_t) > self._idle_timeout_s:
                    await self.end_stream(nonce)
                    closed += 1
        except Exception:
            pass
        return closed


__all__ = ["StreamManager", "StreamContext"]
