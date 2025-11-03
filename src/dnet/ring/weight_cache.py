"""Weight cache with windowed GPU residency and LRU eviction."""

import threading
import time
from typing import Dict, List, Optional
from concurrent.futures import Future

import mlx.core as mx

from ..utils.layer_manager import LayerManager
from ..utils.model import ModelMetadata
from ..utils.logger import logger


class WeightCache:
    """Layer weight cache with windowed GPU residency and LRU eviction.

    Only up to `window_size` layers are kept resident in device memory. All
    other layers remain memory-mapped on disk and can be preheated via
    `madvise` using the layer manager to reduce page faults on load.
    """

    def __init__(
        self,
        assigned_layers: List[int],
        model_metadata: ModelMetadata,
        window_size: Optional[int] = None,
        prefetch_threads: int = 2,
        tracer=None,
        *,
        resident_windows: int = 2,
        use_mxload_fastpath: bool = False,
        prefetch_mode: str = "off",
    ):
        self.assigned_layers = assigned_layers
        resident_windows = max(1, int(resident_windows)) # Resident budget

        if window_size is not None and window_size > 0:
            self.max_weights = min(
                len(self.assigned_layers), 
                max(1, resident_windows * int(window_size)))
        else:
            self.max_weights = len(self.assigned_layers)
        self.cache = {}  # layer_id -> (data, access_time)
        self.reference_counts: Dict[int, int] = {}  # layer_id -> count
        self.layer_manager = LayerManager(
            model_metadata,
            assigned_layers,
            thread_pool_size=int(prefetch_threads or 2),
            use_mxload_fastpath=bool(use_mxload_fastpath),
            prefetch_mode=prefetch_mode,
        )
        self.lock = threading.Lock()

        if not tracer:
            logger.error("Invalid tracer object passed to WeightCache.")
        self.tracer = tracer

        # Track in-flight materializations so compute can wait on prefetch
        self.loading_futures: Dict[int, Future] = {}
        self.prefetch_futures: Dict[int, Future] = {}
        logger.info("WeightCache resident budget: max_weights=%d", self.max_weights)


    def get_weight(self, layer_id: int, *, inc_ref: bool = False) -> Optional[Dict[str, mx.array]]:
        """Get weight from cache"""
        # First, fast path under lock for cache hit or in-flight load
        with self.lock:
            if layer_id in self.cache:
                data, _ = self.cache[layer_id]
                # refresh LRU timestamp
                self.cache[layer_id] = (data, time.time())
                if inc_ref:
                    self.reference_counts[layer_id] = (
                        self.reference_counts.get(layer_id, 0) + 1
                    )
                return data

        with self.tracer.frame("memory.weights", "cache.search") as f:
            with self.lock:  
                if layer_id in self.cache: 
                    data, _ = self.cache[layer_id]
                    self.cache[layer_id] = (data, time.time()) # refresh LRU timestamp
                    if inc_ref:
                        self.reference_counts[layer_id] = (self.reference_counts.get(layer_id, 0) + 1)
                    logger.debug("Cache hit for layer %s, ref=%d inc=%d",
                                 layer_id, self.reference_counts.get(layer_id, 0), int(inc_ref))
                    return data

                inflight = self.loading_futures.get(layer_id) # If a load is in-flight, wait on it outside the lock
                if inflight is None:
                    need_evict = len(self.cache) >= self.max_weights
                    if need_evict:        # Prepare eviction decision now to avoid overfilling once loaded
                        self._evict_lru() # Evict under lock, then proceed to load
                    fut = Future()        # Install a new future marker so others wait
                    self.loading_futures[layer_id] = fut
                    inflight = fut
                    creator = True
                else:
                    creator = False

        if creator: # Perform the blocking load without holding the cache lock
            with self.tracer.frame("memory.weights", "cache.load") as f:
                try:
                    data = self.layer_manager.load_layer_to_gpu(layer_id)
                    f.event("load")

                    try: # Estimate bytes by summing tensor sizes for the layer
                        winfo = self.layer_manager.weight_info.get(layer_id, {})
                        total_bytes = sum(w.size_bytes for w in winfo.values())
                        f.set("bytes", total_bytes)
                    except Exception:
                        total_bytes = 0

                    with self.lock: # Commit to cache under lock
                        self.cache[layer_id] = (data, time.time())
                        if inc_ref:
                            self.reference_counts[layer_id] = (self.reference_counts.get(layer_id, 0) + 1)
                        else:
                            self.reference_counts.setdefault(layer_id, 0)

                        try: # Resolve future and clear from tracking
                            fut = self.loading_futures.pop(layer_id, None)
                            if fut is not None and not fut.done():
                                fut.set_result(True)
                        except Exception:
                            self.loading_futures.pop(layer_id, None)
                    return data

                except Exception as e:
                    with self.lock: # Signal failure to any waiters
                        try:
                            fut = self.loading_futures.pop(layer_id, None)
                            if fut is not None and not fut.done():
                                fut.set_exception(e)
                        except Exception:
                            self.loading_futures.pop(layer_id, None)
                    logger.exception("Failed to load weight %s: %s", layer_id, e)
                    return None

        else: # Not the creator: wait for the in-flight load to complete
            with self.tracer.frame("memory.weights", "cache.wait") as f:
                try:
                    inflight.result()  # block until the creator completes
                except Exception as e:
                    logger.error("Wait for layer %s load failed: %s", layer_id, e)
                    return None

                with self.lock: # Return from cache 
                    data, _ = self.cache.get(layer_id, (None, 0.0))  # type: ignore[assignment]
                    if data is None:
                        logger.error("Wait for layer %s load failed: data not in cache", layer_id)
                        return None

                    self.cache[layer_id] = (data, time.time())
                    if inc_ref:
                        self.reference_counts[layer_id] = (self.reference_counts.get(layer_id, 0) + 1)
                    else:
                        self.reference_counts.setdefault(layer_id, 0)
                    # Resolve future and clear from tracking
                    try:
                        fut = self.loading_futures.pop(layer_id, None)
                        if fut is not None and not fut.done():
                            fut.set_result(True)
                    except Exception:
                        self.loading_futures.pop(layer_id, None)
                logger.info(
                    "[PROFILE][MATERIALIZE] layer=%s ms=%.2f bytes=%.2fMB",
                    layer_id,
                    dt_ms,
                    (total_bytes / 1_048_576),
                )
                return data
            except Exception as e:
                # Signal failure to any waiters
                with self.lock:
                    try:
                        fut = self.loading_futures.pop(layer_id, None)
                        if fut is not None and not fut.done():
                            fut.set_exception(e)
                    except Exception:
                        self.loading_futures.pop(layer_id, None)
                logger.exception("Failed to load weight %s: %s", layer_id, e)
                return None
        else:
            # Not the creator: wait for the in-flight load to complete
            t0w = time.perf_counter()
            try:
                inflight.result()  # block until the creator completes
            except Exception as e:
                logger.error("Wait for layer %s load failed: %s", layer_id, e)
                return None
            wait_ms = (time.perf_counter() - t0w) * 1000.0
            logger.info("[PROFILE][WAIT-WEIGHT] layer=%s ms=%.2f", layer_id, wait_ms)
            # Return from cache (now populated) and update ref/LRU
            with self.lock:
                data, _ = self.cache.get(layer_id, (None, 0.0))  # type: ignore[assignment]
                if data is None:
                    return None
                self.cache[layer_id] = (data, time.time())
                if inc_ref:
                    self.reference_counts[layer_id] = (
                        self.reference_counts.get(layer_id, 0) + 1
                    )
                else:
                    self.reference_counts.setdefault(layer_id, 0)
                return data

    def decrease_reference(self, layer_id: int):
        """Decrease reference count for layer"""
        with self.lock:
            if layer_id in self.reference_counts:
                self.reference_counts[layer_id] -= 1

    def prefetch_to_ram(self, layer_id: int):
        """Asynchronously hint the OS to prefetch layer pages into RAM.

        This does not allocate device memory; it only warms the file-backed
        pages to speed up subsequent `get_weight` loads.
        """
        try:
            # Respect config: skip entirely if prefetch is off
            if self.layer_manager._prefetch_mode == "off":
                return None
            # Avoid spamming prefetch for the same layer; reuse in-flight
            f = self.prefetch_futures.get(layer_id)
            if f is not None and not f.done():
                return f
            f = self.layer_manager.async_prefetch(layer_id)
            self.prefetch_futures[layer_id] = f
            return f
        except Exception:
            return None


    def cancel_all_prefetch(self):
        """Cancel any in-flight prefetch tasks and clear tracking."""
        with self.lock:
            for _, fut in list(self.prefetch_futures.items()):
                try:
                    if fut is not None and not fut.done():
                        fut.cancel()
                except Exception:
                    pass
            self.prefetch_futures.clear()


    def _evict_lru(self):
        """Evict least recently used weight with zero references"""
        candidates = [
            (lid, access_time)
            for lid, (_, access_time) in self.cache.items()
            if self.reference_counts.get(lid, 0) == 0
        ]

        if candidates:
            # Sort by access time, evict oldest
            candidates.sort(key=lambda x: x[1])
            layer_id = candidates[0][0]

            # Hint OS we no longer need these pages in RAM
            try:
                self.layer_manager.release_layer(layer_id)
            except Exception:
                pass

            # Close mmap and remove from cache
            del self.cache[layer_id]
            if layer_id in self.reference_counts:
                del self.reference_counts[layer_id]

            logger.info("Evicted layer %s from cache", layer_id)

    def evict_layer(self, layer_id: int) -> bool:
        """Proactively evict a specific layer if it has no active references.

        Returns True if evicted, False otherwise.
        """
        with self.lock:
            if self.reference_counts.get(layer_id, 0) != 0:
                return False
            if layer_id not in self.cache:
                return True
            try:
                self.layer_manager.release_layer(layer_id)
            except Exception:
                pass
            del self.cache[layer_id]
            if layer_id in self.reference_counts:
                del self.reference_counts[layer_id]
            logger.debug("Proactively evicted layer %s from cache", layer_id)
            return True

    def evict_layers(self, layer_ids: List[int]) -> int:
        """Evict a set of layers if possible; returns count evicted."""
        count = 0
        for lid in layer_ids:
            try:
                if self.evict_layer(lid):
                    count += 1
            except Exception:
                continue
        return count

    def get_resident_layers(self) -> List[int]:
        """Return the current resident layer ids ordered by recency.

        Ordering is from oldest to newest based on last access time. This allows
        callers to keep the most recently used tail and evict the least recent head.
        """
        with self.lock:
            items = sorted(self.cache.items(), key=lambda kv: kv[1][1])
            return [lid for lid, _ in items]
