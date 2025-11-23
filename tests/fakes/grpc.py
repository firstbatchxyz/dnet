"""gRPC server/channel/stub fakes for tests."""

from __future__ import annotations

from typing import Any
import types


class FakeGrpcServer:
    """Minimal grpc.aio server fake for lifecycle tests."""

    def __init__(self):
        self.added: str | None = None
        self.started: bool = False
        self.stopped: int | None = None
        self.handlers: Any = None
        self._reg: Any = None

    def add_insecure_port(self, addr: str):
        self.added = addr

    async def start(self):
        self.started = True

    async def stop(self, grace: int = 0):
        self.stopped = grace

    def add_generic_rpc_handlers(self, handlers):
        self.handlers = handlers

    def add_registered_method_handlers(self, service_name, method_handlers):
        self._reg = (service_name, method_handlers)


class FakeChannel:
    """Minimal gRPC channel with close() for tests that patch channel creation."""

    def __init__(self, addr: str):
        self.addr = addr
        self.closed: bool = False

    async def close(self):
        self.closed = True


class FakeRingStub:
    """Stub of DnetRingServiceStub with configurable MeasureLatency response."""

    ml_success: bool = True
    ml_message: str = "ok"
    ml_node_id: str = "stub"

    def __init__(self, ch: FakeChannel):
        self._ch = ch

    def StreamActivations(self, it):
        async def gen():
            if False:
                yield None

        return gen()

    async def MeasureLatency(self, req):
        return types.SimpleNamespace(
            success=bool(self.ml_success),
            message=str(self.ml_message),
            node_id=str(self.ml_node_id),
        )


class FakeApiStub:
    """Stub of ShardApiServiceStub for SendToken."""

    def __init__(self, ch: FakeChannel):
        self._ch = ch

    async def SendToken(self, req, timeout: float = 3.0):
        return types.SimpleNamespace(success=True, message="ok")
