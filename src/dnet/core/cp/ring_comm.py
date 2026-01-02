"""Ring communication primitives for context parallelism.

Provides async send/recv operations for passing data between CP ranks in a ring topology.
Uses gRPC for transport, with optional overlap of send/recv to hide latency.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable

from grpc import aio as aio_grpc

from dnet.utils.grpc_config import GRPC_AIO_OPTIONS
from dnet.utils.logger import logger


@dataclass
class RingNeighbors:
    """Addresses of neighboring ranks in the ring."""

    prev_address: str  # host:port of rank (id - 1) % N
    next_address: str  # host:port of rank (id + 1) % N


class CPRingCommunicator:
    """
    Manages ring communication for context parallelism.

    Provides async send_recv operation that simultaneously sends to next rank
    and receives from previous rank, enabling pipelined communication.
    """

    def __init__(
        self,
        rank_id: int,
        num_ranks: int,
        neighbors: Optional[RingNeighbors] = None,
    ):
        """
        Initialize ring communicator.

        Args:
            rank_id: This rank's ID (0 to num_ranks-1)
            num_ranks: Total number of CP ranks
            neighbors: Addresses of prev/next ranks (can be set later via connect)
        """
        if num_ranks <= 0:
            raise ValueError(f"num_ranks must be positive, got {num_ranks}")
        if not 0 <= rank_id < num_ranks:
            raise ValueError(f"rank_id {rank_id} out of range [0, {num_ranks})")

        self.rank_id = rank_id
        self.num_ranks = num_ranks
        self.prev_rank = (rank_id - 1) % num_ranks
        self.next_rank = (rank_id + 1) % num_ranks

        self._neighbors = neighbors
        self._prev_channel: Optional[aio_grpc.Channel] = None
        self._next_channel: Optional[aio_grpc.Channel] = None

        # Pending receives keyed by tag
        self._pending_recv: dict[str, asyncio.Future[bytes]] = {}

        # Lock to ensure connect is called once
        self._connect_lock = asyncio.Lock()
        self._connected = False

    async def connect(self, neighbors: RingNeighbors) -> None:
        """
        Establish gRPC channels to neighboring ranks.

        Args:
            neighbors: Addresses for prev/next ranks
        """
        async with self._connect_lock:
            if self._connected:
                return

            self._neighbors = neighbors

            # Connect to prev rank (we receive from them)
            if self.num_ranks > 1:
                self._prev_channel = aio_grpc.insecure_channel(
                    neighbors.prev_address, options=GRPC_AIO_OPTIONS
                )
                self._next_channel = aio_grpc.insecure_channel(
                    neighbors.next_address, options=GRPC_AIO_OPTIONS
                )
                logger.debug(
                    "Rank %d: connected to prev=%s, next=%s",
                    self.rank_id,
                    neighbors.prev_address,
                    neighbors.next_address,
                )

            self._connected = True

    async def disconnect(self) -> None:
        """Close gRPC channels."""
        async with self._connect_lock:
            if self._prev_channel:
                await self._prev_channel.close()
                self._prev_channel = None
            if self._next_channel:
                await self._next_channel.close()
                self._next_channel = None
            self._connected = False

    async def send_recv(
        self,
        send_data: bytes,
        tag: str,
        send_fn: Optional[Callable[[bytes, str], Awaitable[None]]] = None,
        recv_fn: Optional[Callable[[str], Awaitable[bytes]]] = None,
    ) -> bytes:
        """
        Simultaneously send to next rank and receive from previous rank.

        This is the core operation for ring attention - overlapping send/recv
        allows pipelining computation with communication.

        Args:
            send_data: Data to send to next rank
            tag: Unique tag for this communication (e.g., "kv_step_0")
            send_fn: Optional custom send function (for testing)
            recv_fn: Optional custom recv function (for testing)

        Returns:
            Data received from previous rank
        """
        if self.num_ranks == 1:
            # Single rank: no communication needed, return own data
            return send_data

        # Use provided functions or defaults
        do_send = send_fn if send_fn is not None else self._send_to_next
        do_recv = recv_fn if recv_fn is not None else self._recv_from_prev

        # Launch send and recv concurrently using gather
        _, recv_data = await asyncio.gather(
            do_send(send_data, tag),
            do_recv(tag),
        )

        return recv_data

    async def _send_to_next(self, data: bytes, tag: str) -> None:
        """
        Send data to next rank in the ring via gRPC.

        Uses CPRingService.SendBlock unary RPC with raw bytes in a CPBlockFrame.
        """
        if not self._next_channel:
            raise RuntimeError("Not connected to next rank")

        from dnet.protos import dnet_cp_pb2, dnet_cp_pb2_grpc

        stub = dnet_cp_pb2_grpc.CPRingServiceStub(self._next_channel)
        frame = dnet_cp_pb2.CPBlockFrame(
            nonce=tag,
            source_rank=self.rank_id,
            # Use partial_output to carry raw bytes (reusing existing proto field)
            partial_output=dnet_cp_pb2.PartialOutput(output_data=data),
        )

        try:
            ack = await stub.SendBlock(frame)
            if not ack.accepted:
                raise RuntimeError(f"Block rejected by next rank: {ack.error_message}")
            logger.debug(
                "Rank %d: sent %d bytes to rank %d (tag=%s)",
                self.rank_id,
                len(data),
                self.next_rank,
                tag,
            )
        except Exception as e:
            logger.error("Rank %d: failed to send to next rank: %s", self.rank_id, e)
            raise

    async def _recv_from_prev(self, tag: str) -> bytes:
        """
        Receive data from previous rank in the ring.

        Uses a pending receive pattern - the gRPC server calls resolve_recv
        when data arrives, and this method waits on the future.
        """
        if not self._prev_channel:
            raise RuntimeError("Not connected to previous rank")

        # Create a future for this tag if it doesn't exist
        if tag not in self._pending_recv:
            self._pending_recv[tag] = asyncio.get_event_loop().create_future()

        # Wait for the data to arrive (set by resolve_recv when server receives it)
        try:
            data = await asyncio.wait_for(self._pending_recv[tag], timeout=30.0)
            logger.debug(
                "Rank %d: received %d bytes from rank %d (tag=%s)",
                self.rank_id,
                len(data),
                self.prev_rank,
                tag,
            )
            return data
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Rank {self.rank_id}: timeout waiting for data from prev rank (tag={tag})"
            )

    def resolve_recv(self, tag: str, data: bytes) -> None:
        """
        Resolve a pending receive with incoming data.

        Called by the gRPC server when data arrives from prev rank.
        """
        if tag in self._pending_recv:
            self._pending_recv[tag].set_result(data)
            del self._pending_recv[tag]


class MockRingCommunicator:
    """
    Mock ring communicator for testing without actual gRPC.

    Simulates a ring of N ranks where each rank's send_data
    becomes the next rank's recv_data.
    """

    def __init__(self, num_ranks: int):
        """Create a mock ring with num_ranks participants."""
        self.num_ranks = num_ranks
        self._buffers: dict[int, dict[str, bytes]] = {i: {} for i in range(num_ranks)}
        self._lock = asyncio.Lock()

    def get_communicator(self, rank_id: int) -> "MockRankCommunicator":
        """Get a communicator instance for a specific rank."""
        return MockRankCommunicator(self, rank_id, self.num_ranks)


class MockRankCommunicator:
    """Per-rank mock communicator that shares state with the ring."""

    def __init__(self, ring: MockRingCommunicator, rank_id: int, num_ranks: int):
        self._ring = ring
        self.rank_id = rank_id
        self.num_ranks = num_ranks
        self.prev_rank = (rank_id - 1) % num_ranks
        self.next_rank = (rank_id + 1) % num_ranks

    async def send_recv(self, send_data: bytes, tag: str) -> bytes:
        """
        Mock send/recv that stores data for next rank to read.

        In the mock, we store send_data in next_rank's buffer,
        and read from our own buffer (populated by prev_rank).
        """
        if self.num_ranks == 1:
            return send_data

        async with self._ring._lock:
            # Store data for next rank to receive
            self._ring._buffers[self.next_rank][tag] = send_data

        # Small delay to simulate network
        await asyncio.sleep(0.001)

        # Wait for data from prev rank
        for _ in range(100):  # Max 100ms wait
            async with self._ring._lock:
                if tag in self._ring._buffers[self.rank_id]:
                    data = self._ring._buffers[self.rank_id].pop(tag)
                    return data
            await asyncio.sleep(0.001)

        raise TimeoutError(f"Rank {self.rank_id}: timeout waiting for {tag}")
