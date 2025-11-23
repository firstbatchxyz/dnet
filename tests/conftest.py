"""Shared test fixtures and helpers to reduce duplication."""

import asyncio
import typing as _t
import pytest


@pytest.fixture
def wait_until():
    """Async poller to replace arbitrary sleeps in tests.

    Usage:
      await wait_until(lambda: condition(), timeout=0.5)
    """

    async def _wait(
        cond: _t.Callable[[], bool], timeout: float = 0.5, interval: float = 0.01
    ) -> bool:
        deadline = asyncio.get_event_loop().time() + float(timeout)
        while True:
            try:
                if cond():
                    return True
            except Exception:
                # Treat exceptions as not-ready; keep polling
                pass
            if asyncio.get_event_loop().time() >= deadline:
                return False
            await asyncio.sleep(interval)

    return _wait


@pytest.fixture
def patch_http_grpc_client(monkeypatch):
    """Patch shard HTTP gRPC client calls to use FakeChannel/FakeRingStub.

    Returns a function that accepts a dict to capture the last dialed address.
    """

    def _apply(seen_addr: dict):
        from tests.fakes import FakeChannel, FakeRingStub

        def _fake_ch(address: str):
            seen_addr["addr"] = address
            return FakeChannel(address)

        monkeypatch.setattr(
            "dnet.shard.http_api.aio_grpc.insecure_channel", _fake_ch, raising=True
        )
        monkeypatch.setattr(
            "dnet.shard.http_api.DnetRingServiceStub",
            lambda ch: FakeRingStub(ch),
            raising=True,
        )
        return seen_addr

    return _apply


@pytest.fixture
def shard_http_server():
    """Create a Shard HTTPServer + FakeShard with routes set up.

    Provides (srv, shard) and ensures consistent instance name 'S1'.
    """
    import asyncio
    from tests.fakes import FakeShard, FakeDiscovery
    from dnet.shard.http_api import HTTPServer

    shard = FakeShard(1)
    disc = FakeDiscovery({})
    setattr(disc, "instance_name", lambda: "S1")
    srv = HTTPServer(http_port=0, grpc_port=9001, shard=shard, discovery=disc)
    asyncio.run(srv._setup_routes())
    return srv, shard


@pytest.fixture
def patch_shard_hypercorn_serve(monkeypatch):
    """Patch shard HTTPServer's hypercorn serve entry with a controllable fake.

    Returns a dict started={'ok': bool} and the fake function is installed.
    """
    started = {"ok": False}

    async def fake_serve(app, config, shutdown_trigger):
        started["ok"] = True
        fut = shutdown_trigger()
        try:
            await fut
        except Exception:
            pass
        return

    monkeypatch.setattr(
        "dnet.shard.http_api.aio_hypercorn.serve", fake_serve, raising=True
    )
    return started


@pytest.fixture
def patch_api_hypercorn_serve(monkeypatch):
    """Patch API HTTPServer's hypercorn serve entry with a controllable fake.

    Returns a dict started={'ok': bool} and the fake function is installed.
    """
    started = {"ok": False}

    async def fake_serve(app, config, shutdown_trigger):
        started["ok"] = True
        fut = shutdown_trigger()
        try:
            await fut
        except Exception:
            pass
        return

    monkeypatch.setattr(
        "dnet.api.http_api.aio_hypercorn.serve", fake_serve, raising=True
    )
    return started


@pytest.fixture
def patch_ring_grpc_client_ok(monkeypatch):
    """Patch RingAdapter gRPC client to use FakeChannel/FakeRingStub and capture dialed address."""

    def _apply(seen_addr: dict):
        from tests.fakes import FakeChannel, FakeRingStub

        monkeypatch.setattr(
            "dnet.shard.adapters.ring.aio_grpc.insecure_channel",
            lambda addr: (seen_addr.__setitem__("addr", addr) or FakeChannel(addr)),
            raising=True,
        )
        monkeypatch.setattr(
            "dnet.shard.adapters.ring.DnetRingServiceStub",
            lambda ch: FakeRingStub(ch),
            raising=True,
        )
        return seen_addr

    return _apply
