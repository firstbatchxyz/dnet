"""Tests: StreamManager backpressure disable/enable and idle cleanup."""

import asyncio
import pytest

from dnet.core.stream_manager import StreamManager
from tests.fakes import FakeStreamAck, FakeStreamCall


pytestmark = [pytest.mark.core]


def test_backpressure_disables_and_reenables():
    sm = StreamManager(idle_timeout_s=10.0, backoff_s=0.05)

    async def main():
        def factory(_iter):  # ack reader should process one backpressure message
            return FakeStreamCall(
                [FakeStreamAck(accepted=False, message="backpressure: slow")]
            )

        ctx = await sm.get_or_create_stream("n1", factory)
        assert ctx is not None and ctx.open is True

        await asyncio.sleep(0)  # NOTE: only runs if 0, fix later
        assert ctx.disabled is True and ctx.disabled_until > 0

        ctx2 = await sm.get_or_create_stream("n1", factory)
        assert ctx2 is ctx and ctx2.disabled is True

        await asyncio.sleep(
            0.06
        )  # disabled flag should reset when accessing the context
        ctx3 = await sm.get_or_create_stream("n1", factory)
        assert ctx3 is ctx and ctx3.disabled is False

        await sm.end_stream("n1")

    asyncio.run(main())


def test_cleanup_idle_streams_closes_context():
    sm = StreamManager(idle_timeout_s=0.01, backoff_s=0.01)

    async def main():
        def factory(_iter):
            return FakeStreamCall([])

        ctx = await sm.get_or_create_stream("n2", factory)
        assert ctx is not None
        loop = asyncio.get_running_loop()
        sm.get_ctx("n2").last_activity_t = (
            loop.time() - 1.0
        )  # force last_activity_t sufficiently in the past
        closed = await sm.cleanup_idle_streams()
        assert closed == 1
        assert sm.get_ctx("n2") is None

    asyncio.run(main())
