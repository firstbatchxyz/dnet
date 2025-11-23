"""Model metadata and ring model fakes."""

from __future__ import annotations

import types


class FakeWeightSize:
    def __init__(self, size: int):
        self.size_bytes = int(size)


class FakeModelMetadata:
    """Minimal model metadata for tests (subset of real ModelMetadata).

    Provides the attributes commonly accessed in tests and loader helpers:
    - weight_info (per-layer weights)
    - embed_tokens, lm_head, norm (API-layer tensors)
    - model_type, model_config, num_layers, path
    """

    def __init__(
        self,
        weight_info=None,
        *,
        embed_tokens=None,
        lm_head=None,
        norm=None,
        model_type: str = "llama",
        model_config=None,
        num_layers: int = 4,
        path: str | None = None,
    ):
        self.weight_info = weight_info or {}
        self.embed_tokens = embed_tokens or {}
        self.lm_head = lm_head or {}
        self.norm = norm or {}
        self.model_type = model_type
        self.model_config = model_config or {}
        self.num_layers = int(num_layers)
        self.path = path or "/src"


class FakeRingModel:
    """Simple ring model for load/weight binding tests."""

    def __init__(
        self,
        *,
        quant_applies: bool = True,
        tie_word_embeddings: bool = False,
        hidden_size=None,
        vocab_size=None,
    ):
        self.config = types.SimpleNamespace(
            tie_word_embeddings=bool(tie_word_embeddings),
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        self._quant_applies = bool(quant_applies)
        self.eval_called = False
        self.loaded = {}

    def apply_quantization_from_config(self, cfg, model_metadata=None):
        return self._quant_applies

    def eval(self):
        self.eval_called = True

    def load_weights(self, items, strict: bool = False):
        try:
            self.loaded.update(dict(items))
        except Exception:
            pass
