"""Tests for context parallelism ring communication."""

from __future__ import annotations

import asyncio
import pytest

from dnet.core.cp.ring_comm import (
    CPRingCommunicator,
    RingNeighbors,
    MockRingCommunicator,
)


class TestCPRingCommunicator:
    """Tests for the CPRingCommunicator class."""

    def test_init_valid(self):
        """Should initialize with valid rank/num_ranks."""
        comm = CPRingCommunicator(rank_id=0, num_ranks=4)
        assert comm.rank_id == 0
        assert comm.num_ranks == 4
        assert comm.prev_rank == 3
        assert comm.next_rank == 1

    def test_init_middle_rank(self):
        """Should compute correct neighbors for middle rank."""
        comm = CPRingCommunicator(rank_id=2, num_ranks=4)
        assert comm.prev_rank == 1
        assert comm.next_rank == 3

    def test_init_last_rank(self):
        """Should wrap around for last rank."""
        comm = CPRingCommunicator(rank_id=3, num_ranks=4)
        assert comm.prev_rank == 2
        assert comm.next_rank == 0

    def test_init_invalid_num_ranks(self):
        """Should raise on invalid num_ranks."""
        with pytest.raises(ValueError, match="num_ranks must be positive"):
            CPRingCommunicator(rank_id=0, num_ranks=0)

    def test_init_invalid_rank_id(self):
        """Should raise on out-of-range rank_id."""
        with pytest.raises(ValueError, match="rank_id .* out of range"):
            CPRingCommunicator(rank_id=5, num_ranks=4)

    def test_send_recv_single_rank(self):
        """Single rank should return its own data."""

        async def _test():
            comm = CPRingCommunicator(rank_id=0, num_ranks=1)
            data = b"test_data"
            result = await comm.send_recv(data, "tag1")
            assert result == data

        asyncio.run(_test())

    def test_connect_sets_flag(self):
        """Connect should set the connected flag."""

        async def _test():
            comm = CPRingCommunicator(rank_id=0, num_ranks=2)
            neighbors = RingNeighbors(
                prev_address="localhost:50001",
                next_address="localhost:50002",
            )
            await comm.connect(neighbors)
            assert comm._connected
            await comm.disconnect()
            assert not comm._connected

        asyncio.run(_test())


class TestMockRingCommunicator:
    """Tests for the mock ring communicator."""

    def test_two_rank_exchange(self):
        """Two ranks should exchange data correctly."""

        async def _test():
            ring = MockRingCommunicator(num_ranks=2)
            rank0 = ring.get_communicator(0)
            rank1 = ring.get_communicator(1)

            # Run both send_recv concurrently
            data0 = b"from_rank_0"
            data1 = b"from_rank_1"

            results = await asyncio.gather(
                rank0.send_recv(data0, "step1"),
                rank1.send_recv(data1, "step1"),
            )

            # rank0 receives from rank1 (prev of 0 is 1 in 2-rank ring)
            # rank1 receives from rank0 (prev of 1 is 0)
            assert results[0] == data1  # rank0 got data1
            assert results[1] == data0  # rank1 got data0

        asyncio.run(_test())

    def test_four_rank_ring(self):
        """Four ranks should form a proper ring."""

        async def _test():
            ring = MockRingCommunicator(num_ranks=4)
            ranks = [ring.get_communicator(i) for i in range(4)]

            # Each rank sends its ID as bytes
            data = [f"rank_{i}".encode() for i in range(4)]

            results = await asyncio.gather(
                *[ranks[i].send_recv(data[i], "step1") for i in range(4)]
            )

            # Each rank should receive from its previous rank
            # rank 0 receives from rank 3, rank 1 from rank 0, etc.
            for i in range(4):
                prev = (i - 1) % 4
                assert results[i] == data[prev]

        asyncio.run(_test())

    def test_multiple_steps(self):
        """Ring should work across multiple communication steps."""

        async def _test():
            ring = MockRingCommunicator(num_ranks=3)
            ranks = [ring.get_communicator(i) for i in range(3)]

            # Step 1
            data_step1 = [b"s1_r0", b"s1_r1", b"s1_r2"]
            results1 = await asyncio.gather(
                *[ranks[i].send_recv(data_step1[i], "step1") for i in range(3)]
            )

            # Step 2: use results from step 1
            results2 = await asyncio.gather(
                *[ranks[i].send_recv(results1[i], "step2") for i in range(3)]
            )

            # After 2 steps in a 3-rank ring, data has rotated 2 positions
            # rank 0: recv from 2, who recv'd from 1 -> original rank 1 data
            assert results2[0] == b"s1_r1"
            assert results2[1] == b"s1_r2"
            assert results2[2] == b"s1_r0"

        asyncio.run(_test())

    def test_single_rank_mock(self):
        """Single rank mock should return own data."""

        async def _test():
            ring = MockRingCommunicator(num_ranks=1)
            rank0 = ring.get_communicator(0)

            data = b"solo"
            result = await rank0.send_recv(data, "tag")
            assert result == data

        asyncio.run(_test())


class TestRingNeighbors:
    """Tests for the RingNeighbors dataclass."""

    def test_creation(self):
        """Should create RingNeighbors with addresses."""
        neighbors = RingNeighbors(
            prev_address="192.168.1.1:50051",
            next_address="192.168.1.2:50051",
        )
        assert neighbors.prev_address == "192.168.1.1:50051"
        assert neighbors.next_address == "192.168.1.2:50051"
