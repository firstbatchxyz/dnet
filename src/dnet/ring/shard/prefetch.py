from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from typing import Dict
from threading import Lock
from queue import Queue

import mlx.core as mx

from dnet.ring.weight_cache import WeightCache

from ...utils.logger import logger


class PrefetchMixin:
    _mlx_lock: Lock
    _prefetch_scheduled: set[int]
    _prefetch_pending: set[int]
    _prefetch_pause: asyncio.Event
    _profile: bool
    node_id: int
    running: bool
    weight_cache: WeightCache
    weight_prefetch_queue: Queue[int]
    _materialize_prefetch_default: bool
    executor: ThreadPoolExecutor
    _touch_during_compute: bool
    _compute_busy: asyncio.Event

    def _touch_weights(self, layer_id: int, weights: Dict[str, mx.array]) -> None:
        mode = getattr(self, "_prefetch_touch_mode", "none")
        if mode in ("", "none"):
            return

        async_flag = bool(getattr(self, "_prefetch_async", True))
        frac = float(getattr(self, "_prefetch_fraction", 0.25))
        frac = max(0.0, min(1.0, frac))
        budget_ms = float(getattr(self, "_prefetch_budget_ms", 0.0))

        ops: list[mx.array] = []
        total_bytes = 0
        t0 = time.perf_counter()
        for v in weights.values():
            try:
                total_bytes += int(v.size) * int(v.dtype.size)
            except Exception:
                pass
            try:
                if mode == "sum":
                    ops.append(mx.sum(v))
                elif mode == "stripe":
                    step = 1 if frac >= 1.0 else max(1, int(1.0 / max(1e-6, frac)))
                    sl = v[(..., slice(None, None, step))]
                    ops.append(mx.sum(sl))
            except Exception:
                continue

        if budget_ms > 0.0 and (time.perf_counter() - t0) * 1000.0 > budget_ms:
            return

        try:
            if ops:
                with self._mlx_lock:
                    if async_flag:
                        mx.async_eval(*ops)
                    else:
                        mx.eval(*ops)
                if self._profile:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    logger.info(
                        "[PROFILE][PREFETCH-TOUCH] node=%s layer=%d mode=%s ops=%d bytes=%.1fMB ms=%.2f async=%d",
                        self.node_id,
                        layer_id,
                        mode,
                        len(ops),
                        total_bytes / 1_048_576,
                        dt_ms,
                        int(async_flag),
                    )
        except Exception:
            pass

    def _prefetch_to_ram(self, layer_id: int):
        if layer_id not in self._prefetch_scheduled:
            self._prefetch_scheduled.add(layer_id)
            fio = bool(getattr(self, "_file_io_direct", False))
            if fio or getattr(self, "_resident_windows", 2) <= 1:
                return
            if self._prefetch_pause.is_set():
                try:
                    self._prefetch_pending.add(layer_id)
                except Exception:
                    pass
            else:
                self.weight_cache.prefetch_to_ram(layer_id)

    def _enqueue_weight_prefetch(self, layer_id: int):
        try:
            if self._prefetch_pause.is_set():
                try:
                    self._prefetch_pending.add(layer_id)
                except Exception:
                    pass
                return
            self.weight_prefetch_queue.put(layer_id, timeout=0.01)
        except Exception:
            pass

    async def _prefetch_worker(self):
        while self.running:
            try:
                if self._prefetch_pause.is_set():
                    await asyncio.sleep(0.01)
                    continue
                batch: list[int] = []
                try:
                    for _ in range(8):
                        batch.append(self.weight_prefetch_queue.get_nowait())
                except Exception:
                    pass
                if not batch:
                    layer_id = self.weight_prefetch_queue.get_nowait()
                    batch = [layer_id]

                _mat_pref = bool(getattr(self, "_materialize_prefetch_default", False))

                try:
                    self._prefetch_active += len(batch)
                    for layer_id in batch:
                        if _mat_pref:
                            weights = await asyncio.get_running_loop().run_in_executor(
                                self.executor,
                                lambda lid=layer_id: self.weight_cache.get_weight(
                                    lid, inc_ref=False
                                ),
                            )
                            try:
                                if isinstance(weights, dict):
                                    busy = self._compute_busy.is_set()
                                    if (not busy) or self._touch_during_compute:
                                        self._touch_weights(layer_id, weights)
                                    else:
                                        self._prefetch_pending.add(layer_id)
                                        if getattr(self, "_profile", False):
                                            logger.info(
                                                f"[PROFILE][PREFETCH-TOUCH] defer=1 layer={layer_id} busy=1"
                                            )
                            except Exception:
                                pass
                        else:
                            await asyncio.get_running_loop().run_in_executor(
                                self.executor,
                                self.weight_cache.prefetch_to_ram,
                                layer_id,
                            )
                        if logging.getLogger("piped_mlx").isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "Prefetched weights for layer %s (materialize=%s)",
                                layer_id,
                                int(_mat_pref),
                            )
                finally:
                    self._prefetch_active = max(0, self._prefetch_active - len(batch))

            except Exception as e:
                await asyncio.sleep(0.02 if e else 0.005)
