from __future__ import annotations

import asyncio
from .attrib import RingShardNodeAttributes


class PrefetchMixin(RingShardNodeAttributes):

    def _prefetch_to_ram(self, layer_id: int):
        if layer_id not in self._prefetch_scheduled:
            self._prefetch_scheduled.add(layer_id)
            if self._resident_windows <= 1:
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

                try:
                    self._prefetch_active += len(batch)
                    for layer_id in batch:
                        await asyncio.get_running_loop().run_in_executor(
                            self.executor,
                            self.weight_cache.prefetch_to_ram,
                            layer_id,
                        )
                finally:
                    self._prefetch_active = max(0, self._prefetch_active - len(batch))

            except Exception as e:
                await asyncio.sleep(0.02 if e else 0.005)
