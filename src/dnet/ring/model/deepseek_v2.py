from typing import Any, Dict, List, Optional, Tuple, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.deepseek_v2 import DeepseekV2DecoderLayer, ModelArgs

from .base import BaseRingModel


class DeepseekV2RingModel(BaseRingModel):
    model_type = "deepseek_v2"

    def __init__(
        self,
        model_config: Any,
        assigned_layers: Optional[List[int]] = None,
        is_api_layer: bool = False,
        shard_config: Optional[Any] = None,
        model_metadata: Optional[Any] = None,
    ):
        super().__init__()

        if is_api_layer and assigned_layers:
            raise RuntimeError("API layer doesn't handle layers")

        self.model_config = model_config
        self.is_api_layer = is_api_layer
        self.config = config = ModelArgs.from_dict(model_config)

        # The API layer handles embedding and normalization
        if is_api_layer:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Other layers: keep a local, zero-based list and a map abs->local
        self.layers: List[nn.Module] = []
        self.abs_to_local: Dict[int, int] = {}
        for i, layer in enumerate(sorted(assigned_layers or [])):
            self.layers.append(DeepseekV2DecoderLayer(config, layer))
            self.abs_to_local[layer] = i

    def embed(self, x: mx.array) -> mx.array:
        return self.embed_tokens(x) if self.is_api_layer else x

    def normalize(self, x: mx.array) -> mx.array:
        return self.norm(x) if self.is_api_layer else x

    def lm_project(self, x: mx.array) -> mx.array:
        return self.lm_head(x) if self.is_api_layer else x

    def forward(self, x: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        mask = None
        T = cast(tuple[int, ...], x.shape)[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(x.dtype)

        # If provided, expect a per-layer cache aligned with our local layer order
        if cache is None:
            cache_list: List[Any] = [None] * len(self.layers)
        else:
            cache_list = cache

        # Apply layers in local order 0..len-1
        for i, layer in enumerate(self.layers):
            c = cache_list[i] if i < len(cache_list) else None
            x = layer(x, mask, c)

        return x

    def apply_single_layer(
        self, layer_idx: int, x: mx.array, cache: Optional[List[Any]] = None
    ) -> mx.array:
        """Apply a single decoder layer identified by absolute index."""
        if layer_idx not in self.abs_to_local:
            raise RuntimeError(f"Layer {layer_idx} not hosted on this model instance")

        # Create attention mask if needed
        mask = None
        T = cast(tuple[int, ...], x.shape)[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(x.dtype)

        # Map absolute layer index to local cache index
        local_idx = self.abs_to_local[layer_idx]
        c = None
        if cache is not None and local_idx < len(cache):
            c = cache[local_idx]

        layer = self.layers[local_idx]
        return layer(x, mask, c)

    @property
    def decoding_layers(self):
        return self.layers

    @property
    def head_dim(self) -> Tuple[int, int]:
        return (
            self.config.qk_nope_head_dim + self.config.qk_rope_head_dim,
            self.config.v_head_dim,
        )

    @property
    def n_kv_heads(self) -> int:
        return self.config.num_key_value_heads

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    # load_weights inherited from BaseRingModel
