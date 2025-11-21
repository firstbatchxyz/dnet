from __future__ import annotations

from dnet.core.types.messages import ActivationMessage
from dnet.utils.logger import logger
from ..models import ShardLoadModelRequest
from .base import ComputePolicy, register_policy


@register_policy("noop")
class NoopPolicy(ComputePolicy):
    def configure_policy_for_model(self, req: ShardLoadModelRequest) -> None:
        self._mode = "noop"
        self.window_size = 0
        self.weight_cache = None
        logger.info(
            "NoopPolicy configured: compute disabled for shard %s",
            self.runtime.shard_id,
        )

    def process(self, msg: ActivationMessage) -> None:
        try:
            if self.runtime.input_pool:
                self.runtime.input_pool.release(msg.pool_id)
        except Exception:
            pass
        logger.debug("NoopPolicy dropped activation nonce=%s", msg.nonce)

    def clear(self):
        pass
