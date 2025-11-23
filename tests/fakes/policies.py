"""Policy-related fakes used by ShardRuntime/policy tests."""

from __future__ import annotations

from typing import Any, Dict


class FakePolicyPlan:
    def __init__(
        self,
        mode: str = "fit",
        window_size: int = 1,
        resident_windows: int = 1,
        is_sliding: bool = False,
    ):
        self.mode = mode
        self.window_size = int(window_size)
        self.resident_windows = int(resident_windows)
        self.is_sliding = bool(is_sliding)


class FakePolicy:
    def __init__(self, runtime, resident_windows: int):
        self.runtime = runtime
        self.resident_windows = int(resident_windows)
        self.window_size: int = 0
        self.configured: Any = None
        self.processed: bool = False

    def process(self, msg):
        self.configured = self.configured or "process_called"
        self.processed = True

    def configure_policy_for_model(self, req):
        self.configured = req

    def clear(self):
        pass


class FakeComputeModel:
    def __init__(self, mx_mod=None):
        import mlx.core as mx

        self._mx = mx_mod or mx
        self.bound: Dict[str, Any] = {}
        self.unloaded: list[int] = []

    def embed(self, toks):
        try:
            arr = self._mx.asarray(toks)
        except Exception:
            arr = toks
        return arr.astype(self._mx.float32)

    def apply_single_layer(self, lyr, x, cache=None):
        return x + self._mx.ones_like(x) * float(lyr)

    def normalize(self, x):
        return x

    def lm_project(self, x):
        return self._mx.arange(0, 4, dtype=self._mx.float32)

    def load_weights(self, items, strict: bool = False):
        for k, v in dict(items).items():
            self.bound[k] = v

    def unload_layers(self, layers):
        self.unloaded.extend(list(layers))


class FakeSampler:
    def sample(self, logits, config, req_logprobs, req_top_logprobs):
        from .api import FakeTokenResult

        return FakeTokenResult(7, -0.1, {7: -0.1})
