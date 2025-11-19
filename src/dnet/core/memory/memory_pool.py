"""Memory pool management for ring topology."""

import threading
import time
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from dnet.utils.logger import logger
from dnet.core.types.messages import PoolStatus


@dataclass
class BufferInfo:
    """Metadata for a memory pool buffer."""

    buffer_id: int
    size: int
    status: PoolStatus
    last_used: float
    ref_count: int = 0


class DynamicMemoryPool:
    """Dynamic memory pool that allocates buffers of varying sizes."""

    def __init__(self, total_memory_mb: int = 512, min_buffer_size: int = 1024) -> None:
        """Initialize dynamic memory pool.

        Args:
            total_memory_mb: Total pool size in megabytes
            min_buffer_size: Minimum buffer size in bytes
        """
        self.total_memory_bytes = total_memory_mb * 1024 * 1024
        self.min_buffer_size = min_buffer_size
        self.used_memory = 0
        self.next_buffer_id = 0

        # Storage
        self.buffers: Dict[int, mx.array] = {}  # buffer_id -> mlx array
        self.buffer_info: Dict[int, BufferInfo] = {}  # buffer_id -> metadata
        self.size_to_buffers: Dict[int, List[int]] = {}  # size -> [buffer_ids]

        self.lock = threading.Lock()

        logger.info(f"Initialized dynamic memory pool with {total_memory_mb}MB")

    def allocate(self, size_bytes: int, dtype: mx.Dtype) -> Optional[int]:
        """Allocate a buffer of exact size.

        Args:
            size_bytes: Size in bytes
            dtype: MLX dtype

        Returns:
            Buffer ID, or None if allocation failed
        """
        with self.lock:
            # Round up to nearest multiple of dtype size (alignment)
            dsize = dtype.size
            aligned_size = ((size_bytes + (dsize - 1)) // dsize) * dsize

            # Try to find existing free buffer of exact size
            buffer_id = self._find_free_buffer(aligned_size)
            if buffer_id is not None:
                self.buffer_info[buffer_id].status = PoolStatus.ALLOCATED
                self.buffer_info[buffer_id].ref_count = 1
                logger.debug(f"Reused buffer {buffer_id} of size {aligned_size}")
                return buffer_id

            # Check if we have enough memory to allocate new buffer
            if self.used_memory + aligned_size > self.total_memory_bytes:
                # Try to free up memory
                freed = self._evict_unused_buffers(aligned_size)
                if not freed:
                    logger.warning(
                        f"Cannot allocate {aligned_size} bytes - insufficient memory"
                    )
                    return None

            # Allocate new buffer
            try:
                buffer = mx.zeros(aligned_size // dsize, dtype=dtype)
                buffer_id = self.next_buffer_id
                self.next_buffer_id += 1

                # Store buffer and metadata
                self.buffers[buffer_id] = buffer
                self.buffer_info[buffer_id] = BufferInfo(
                    buffer_id=buffer_id,
                    size=aligned_size,
                    status=PoolStatus.ALLOCATED,
                    last_used=time.time(),
                    ref_count=1,
                )

                # Add to size index
                if aligned_size not in self.size_to_buffers:
                    self.size_to_buffers[aligned_size] = []
                self.size_to_buffers[aligned_size].append(buffer_id)

                self.used_memory += aligned_size

                logger.debug(
                    f"Allocated new buffer {buffer_id} of size {aligned_size} bytes"
                )
                return buffer_id

            except MemoryError:
                logger.error(
                    f"Failed to allocate {aligned_size} bytes - system out of memory"
                )
                return None

    def get_buffer(self, buffer_id: int) -> Optional[mx.array]:
        """Get buffer by ID.

        Args:
            buffer_id: Buffer identifier

        Returns:
            MLX array, or None if not found
        """
        with self.lock:
            if buffer_id in self.buffers and buffer_id in self.buffer_info:
                info = self.buffer_info[buffer_id]
                if info.status != PoolStatus.FREE:
                    info.last_used = time.time()
                    return self.buffers[buffer_id]
        return None

    def get_buffer_view(
        self, buffer_id: int, shape: Tuple[int, ...]
    ) -> Optional[mx.array]:
        """Get a shaped view of the buffer.

        Args:
            buffer_id: Buffer identifier
            shape: Desired shape

        Returns:
            Reshaped MLX array, or None if buffer too small
        """
        buffer = self.get_buffer(buffer_id)
        if buffer is not None:
            try:
                required_size = reduce(mul, shape)
                if required_size <= len(buffer):
                    return buffer[:required_size].reshape(shape)
                else:
                    logger.error(f"Buffer {buffer_id} too small for shape {shape}")
            except ValueError as e:
                logger.error(f"Cannot reshape buffer {buffer_id} to {shape}: {e}")
        return None

    def release(self, buffer_id: int) -> None:
        """Decrease reference count and potentially free buffer.

        Args:
            buffer_id: Buffer identifier
        """
        with self.lock:
            if buffer_id in self.buffer_info:
                info = self.buffer_info[buffer_id]
                info.ref_count -= 1

                if info.ref_count <= 0:
                    info.status = PoolStatus.FREE
                    info.ref_count = 0
                    logger.debug(f"Released buffer {buffer_id}")

    def _find_free_buffer(self, size: int) -> Optional[int]:
        """Find a free buffer of exact size.

        Args:
            size: Required size in bytes

        Returns:
            Buffer ID, or None if not found
        """
        if size in self.size_to_buffers:
            for buffer_id in self.size_to_buffers[size]:
                if (
                    buffer_id in self.buffer_info
                    and self.buffer_info[buffer_id].status == PoolStatus.FREE
                ):
                    return buffer_id
        return None

    def _evict_unused_buffers(self, needed_bytes: int) -> bool:
        """Evict unused buffers to free up memory.

        Args:
            needed_bytes: How many bytes we need to free

        Returns:
            True if enough memory was freed
        """
        # Sort free buffers by last used time (LRU)
        free_buffers = [
            (info.last_used, buffer_id, info.size)
            for buffer_id, info in self.buffer_info.items()
            if info.status == PoolStatus.FREE
        ]
        free_buffers.sort()  # Oldest first

        freed_bytes = 0
        for _, buffer_id, size in free_buffers:
            if freed_bytes >= needed_bytes:
                break

            # Remove buffer
            if buffer_id in self.buffers:
                del self.buffers[buffer_id]
            if buffer_id in self.buffer_info:
                # Remove from size index
                buffer_size = self.buffer_info[buffer_id].size
                if buffer_size in self.size_to_buffers:
                    self.size_to_buffers[buffer_size].remove(buffer_id)
                    if not self.size_to_buffers[buffer_size]:
                        del self.size_to_buffers[buffer_size]

                del self.buffer_info[buffer_id]

            freed_bytes += size
            self.used_memory -= size
            logger.debug(f"Evicted buffer {buffer_id}, freed {size} bytes")

        return freed_bytes >= needed_bytes

    def get_stats(self) -> Dict:
        """Get memory pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        with self.lock:
            total_buffers = len(self.buffer_info)
            free_buffers = sum(
                1
                for info in self.buffer_info.values()
                if info.status == PoolStatus.FREE
            )
            allocated_buffers = sum(
                1
                for info in self.buffer_info.values()
                if info.status == PoolStatus.ALLOCATED
            )

            return {
                "total_memory_mb": self.total_memory_bytes // (1024 * 1024),
                "used_memory_mb": self.used_memory // (1024 * 1024),
                "free_memory_mb": (self.total_memory_bytes - self.used_memory)
                // (1024 * 1024),
                "total_buffers": total_buffers,
                "free_buffers": free_buffers,
                "allocated_buffers": allocated_buffers,
                "buffer_sizes": list(self.size_to_buffers.keys()),
            }


class LayerAwareMemoryPool:
    """Memory pool that's aware of layer-specific activation patterns."""

    def __init__(self, total_memory_mb: int = 512) -> None:
        """Initialize layer-aware memory pool.

        Args:
            total_memory_mb: Total pool size in megabytes
        """
        self.pool = DynamicMemoryPool(total_memory_mb)
        self.layer_stats: Dict[int, Dict] = {}  # layer_id -> size statistics
        self.lock = threading.Lock()

    def allocate_for_layer(
        self,
        layer_id: int,
        shape: Tuple[int, ...],
        dtype: mx.Dtype,
    ) -> Optional[int]:
        """Allocate buffer for specific layer output.

        Args:
            layer_id: Layer identifier
            shape: Tensor shape
            dtype: MLX dtype

        Returns:
            Buffer ID, or None if allocation failed
        """
        size_bytes = reduce(mul, shape) * dtype.size

        # Track layer statistics
        with self.lock:
            if layer_id not in self.layer_stats:
                self.layer_stats[layer_id] = {
                    "sizes": [],
                    "shapes": [],
                    "allocations": 0,
                }

            stats = self.layer_stats[layer_id]
            stats["sizes"].append(size_bytes)
            stats["shapes"].append(shape)
            stats["allocations"] += 1

            # Keep only recent statistics
            if len(stats["sizes"]) > 100:
                stats["sizes"] = stats["sizes"][-50:]
                stats["shapes"] = stats["shapes"][-50:]

        buffer_id = self.pool.allocate(size_bytes, dtype)
        if buffer_id is not None:
            logger.debug(
                f"Allocated buffer {buffer_id} for layer {layer_id}, shape {shape}"
            )

        return buffer_id

    def get_layer_buffer(
        self, buffer_id: int, shape: Tuple[int, ...]
    ) -> Optional[mx.array]:
        """Get shaped buffer for layer output.

        Args:
            buffer_id: Buffer identifier
            shape: Desired shape

        Returns:
            Reshaped MLX array, or None if not found
        """
        return self.pool.get_buffer_view(buffer_id, shape)

    def get_typical_size(self, layer_id: int) -> Optional[int]:
        """Get typical activation size for a layer.

        Args:
            layer_id: Layer identifier

        Returns:
            Median size in bytes, or None if no data
        """
        with self.lock:
            if layer_id in self.layer_stats and self.layer_stats[layer_id]["sizes"]:
                sizes = self.layer_stats[layer_id]["sizes"]
                # Return median size
                return sorted(sizes)[len(sizes) // 2]
        return None

    def release(self, buffer_id: int) -> None:
        """Release buffer.

        Args:
            buffer_id: Buffer identifier
        """
        self.pool.release(buffer_id)

    def get_buffer(self, buffer_id: int) -> Optional[mx.array]:
        """Get raw buffer.

        Args:
            buffer_id: Buffer identifier

        Returns:
            MLX array, or None if not found
        """
        return self.pool.get_buffer(buffer_id)

    def get_stats(self) -> Dict:
        """Get comprehensive statistics.

        Returns:
            Dictionary with pool and layer statistics
        """
        pool_stats = self.pool.get_stats()

        with self.lock:
            layer_stats = {}
            for layer_id, stats in self.layer_stats.items():
                if stats["sizes"]:
                    layer_stats[layer_id] = {
                        "allocations": stats["allocations"],
                        "avg_size_mb": sum(stats["sizes"])
                        / len(stats["sizes"])
                        / (1024 * 1024),
                        "recent_shapes": list(
                            set(stats["shapes"][-10:])
                        ),  # Recent unique shapes
                    }

        return {"pool": pool_stats, "layer_stats": layer_stats}
