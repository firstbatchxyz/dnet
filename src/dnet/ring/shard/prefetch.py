from __future__ import annotations

import asyncio
from .attrib import RingShardNodeAttributes


class PrefetchMixin(RingShardNodeAttributes):

    def _prefetch_to_ram(self, layer_id: int):
        try:
            if self.weight_cache:
                self.weight_cache.prefetch_to_ram(layer_id)
        except Exception:
            pass

    def _enqueue_weight_prefetch(self, layer_id: int):
        # No-op in sequential IO; retained for compatibility
        return
