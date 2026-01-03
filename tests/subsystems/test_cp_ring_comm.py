"""Tests for context parallelism ring communication."""

from __future__ import annotations

import asyncio
import pytest

from dnet.core.cp.ring_comm import (
    CPRingCommunicator,
    RingNeighbors,
    start_cp_ring_server,
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


class TestRealGRPCRingCommunication:
    """Tests for ring communication using real gRPC servers."""

    def test_two_rank_exchange(self):
        """Two ranks should exchange data correctly via real gRPC."""

        async def _test():
            base_port = 59200
            num_ranks = 2

            # Create communicators
            comms = [
                CPRingCommunicator(rank_id=i, num_ranks=num_ranks)
                for i in range(num_ranks)
            ]

            # Start gRPC servers
            servers = []
            for i in range(num_ranks):
                server = await start_cp_ring_server(
                    port=base_port + i, communicator=comms[i]
                )
                servers.append(server)

            # Connect to neighbors
            for i in range(num_ranks):
                prev_rank = (i - 1) % num_ranks
                next_rank = (i + 1) % num_ranks
                neighbors = RingNeighbors(
                    prev_address=f"localhost:{base_port + prev_rank}",
                    next_address=f"localhost:{base_port + next_rank}",
                )
                await comms[i].connect(neighbors)

            try:
                # Run both send_recv concurrently
                data0 = b"from_rank_0"
                data1 = b"from_rank_1"

                results = await asyncio.gather(
                    comms[0].send_recv(data0, "step1"),
                    comms[1].send_recv(data1, "step1"),
                )

                # rank0 receives from rank1 (prev of 0 is 1 in 2-rank ring)
                # rank1 receives from rank0 (prev of 1 is 0)
                assert results[0] == data1  # rank0 got data1
                assert results[1] == data0  # rank1 got data0
            finally:
                for comm in comms:
                    await comm.disconnect()
                for server in servers:
                    await server.stop(grace=0.1)

        asyncio.run(_test())

    def test_four_rank_ring(self):
        """Four ranks should form a proper ring via real gRPC."""

        async def _test():
            base_port = 59210
            num_ranks = 4

            # Create communicators
            comms = [
                CPRingCommunicator(rank_id=i, num_ranks=num_ranks)
                for i in range(num_ranks)
            ]

            # Start gRPC servers
            servers = []
            for i in range(num_ranks):
                server = await start_cp_ring_server(
                    port=base_port + i, communicator=comms[i]
                )
                servers.append(server)

            # Connect to neighbors
            for i in range(num_ranks):
                prev_rank = (i - 1) % num_ranks
                next_rank = (i + 1) % num_ranks
                neighbors = RingNeighbors(
                    prev_address=f"localhost:{base_port + prev_rank}",
                    next_address=f"localhost:{base_port + next_rank}",
                )
                await comms[i].connect(neighbors)

            try:
                # Each rank sends its ID as bytes
                data = [f"rank_{i}".encode() for i in range(num_ranks)]

                results = await asyncio.gather(
                    *[comms[i].send_recv(data[i], "step1") for i in range(num_ranks)]
                )

                # Each rank should receive from its previous rank
                for i in range(num_ranks):
                    prev = (i - 1) % num_ranks
                    assert results[i] == data[prev]
            finally:
                for comm in comms:
                    await comm.disconnect()
                for server in servers:
                    await server.stop(grace=0.1)

        asyncio.run(_test())

    def test_single_rank(self):
        """Single rank should return own data (no gRPC needed)."""

        async def _test():
            comm = CPRingCommunicator(rank_id=0, num_ranks=1)
            data = b"solo"
            result = await comm.send_recv(data, "tag")
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
