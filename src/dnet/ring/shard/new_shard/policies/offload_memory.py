from .base import register_policy, ComputePolicy

@register_policy("offload")
@register_policy("sliding_fit")
class OffloadingPolicy(ComputePolicy):
    def configure_policy_for_model(self, req):
        pass
        # look at runtime.compute_config.mode to decide offload vs sliding
        # set self._mode, window_size, resident_windows
        # create WeightCache with small resident_windows, fastpath, etc.

    def process(self, msg):
        pass
        # basically current _process_activation, including:
        # - _prepared_by_nonce waits
        # - sliding_fit branches when self._mode == "sliding_fit"
        # - evict windows aggressively when offload
