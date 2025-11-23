"""Stream-related fakes for testing StreamManager and streaming RPCs."""

from __future__ import annotations

from typing import List


class FakeStreamAck:
    """Simple ACK object with attributes used by StreamManager."""

    def __init__(self, *, accepted: bool = True, message: str = ""):
        self.accepted = bool(accepted)
        self.message = str(message)


class FakeStreamCall:
    """Async-iterable call that yields acks and supports aclose()."""

    def __init__(self, acks: List[FakeStreamAck]):
        self._acks = list(acks)
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._acks:
            return self._acks.pop(0)
        raise StopAsyncIteration

    async def aclose(self):
        self._closed = True
