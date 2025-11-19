from dnet.core.memory.weight_cache import WeightCache
from ..models import ShardLoadModelRequest
from dnet.utils.logger import logger
from .base import register_policy, ComputePolicy
from dnet.utils.repack import ensure_repacked_for_layers
import time
from dnet.utils.model import ModelMetadata, get_model_metadata

@register_policy("offload")
@register_policy("sliding_fit")
class OffloadingPolicy(ComputePolicy):
    def configure_policy_for_model(self, req: ShardLoadModelRequest) -> None:
        rt = self.runtime

        # Repack for offload/sliding_fit
        try:
            t0 = time.perf_counter()
            repacked_dir, did_repack = ensure_repacked_for_layers(
                rt.model_path, rt._assigned_sorted
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            rt.model_path = str(repacked_dir)
            rt.model_metadata = get_model_metadata(rt.model_path)

            rt.compute_config.mxload_fastpath = True
            rt.compute_config.prefetch_mode = "off"

            logger.info(
                "[REPACK] shard=%s dst=%s layers=%s repacked=%s ms=%.1f",
                rt.shard_id,
                rt.model_path,
                len(rt._assigned_sorted),
                int(did_repack),
                dt_ms,
            )
        except Exception as e:
            logger.warning(
                "Runtime %s: repack failed or skipped: %s", rt.shard_id, e
            )

    def after_model_loaded(self) -> None:
        rt = self.runtime
        rt.weight_cache = WeightCache(
            rt.assigned_layers,
            rt.model_metadata,
            window_size=self.window_size,
            prefetch_threads=rt._device_prefetch_workers,
            resident_windows=self._resident_windows,
            use_mxload_fastpath=rt.compute_config.mxload_fastpath,
            prefetch_mode=rt.compute_config.prefetch_mode,
        )

    def process(self, msg):
        pass
        # basically current _process_activation, including:
        # - _prepared_by_nonce waits
        # - sliding_fit branches when self._mode == "sliding_fit"
        # - evict windows aggressively when offload
