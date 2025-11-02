from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache

from .base import BaseRingModel


class GptOssRingModel(BaseRingModel):
    model_type = "gpt_oss"

    def __init__(
        self,
        model_config: Any,
        assigned_layers: Optional[List[int]] = None,
        is_api_layer: bool = False,
        shard_config: Optional[Any] = None,
    ):
        super().__init__()

        if is_api_layer and assigned_layers:
            raise RuntimeError("API layer doesn't handle layers")

        from mlx_lm.models.gpt_oss import ModelArgs, TransformerBlock  # type: ignore

        self.model_config = model_config
        self.is_api_layer = is_api_layer
        self.config = config = ModelArgs.from_dict(model_config)

        self.window_size: int = int(getattr(config, "sliding_window", 0) or 0)
        # Layer types alternate sliding/full by default when not provided
        if getattr(config, "layer_types", None):
            self.layer_types: List[str] = list(config.layer_types)
        else:
            half = max(1, int(config.num_hidden_layers // 2))
            self.layer_types = (["sliding_attention", "full_attention"] * half)[: int(config.num_hidden_layers)]

        # API layer owns embeddings/norm/head when used at endpoints
        if is_api_layer:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.layers: List[nn.Module] = []
        self.abs_to_local: Dict[int, int] = {}
        for i, layer in enumerate(sorted(assigned_layers or [])):
            self.layers.append(TransformerBlock(config))
            self.abs_to_local[layer] = i

        self._cached_mask_state: Optional[int] = None
        self._cached_full_mask = None
        self._cached_swa_mask = None

    def embed(self, x: mx.array) -> mx.array:
        return self.embed_tokens(x) if self.is_api_layer else x

    def normalize(self, x: mx.array) -> mx.array:
        return self.norm(x) if self.is_api_layer else x

    def lm_project(self, x: mx.array) -> mx.array:
        return self.lm_head(x) if self.is_api_layer else x

    def _mask_for_layer(self, abs_idx: int, x: mx.array, c: Any) -> Optional[mx.array]:
        try:
            T = int(x.shape[1]) if len(x.shape) > 1 else 1
        except Exception:
            T = 1
        if T <= 1:
            return None
        lt = self.layer_types[abs_idx] if 0 <= abs_idx < len(self.layer_types) else "full_attention"
        if lt == "sliding_attention" and self.window_size > 0:
            return create_attention_mask(x, c, window_size=self.window_size)
        return create_attention_mask(x, c)

    def forward(self, x: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        if cache is None:
            cache = [None] * len(self.layers)
        # Apply local layers in absolute index order
        for abs_idx, local_idx in sorted(self.abs_to_local.items()):
            c = cache[local_idx] if local_idx < len(cache) else None
            mask = self._mask_for_layer(abs_idx, x, c)
            x = self.layers[local_idx](x, mask, c)
        return x

    def apply_single_layer(
        self, layer_idx: int, x: mx.array, cache: Optional[List[Any]] = None
    ) -> mx.array:
        if layer_idx not in self.abs_to_local:
            raise RuntimeError(f"Layer {layer_idx} not hosted on this model instance")
        local_idx = self.abs_to_local[layer_idx]
        c = None
        if cache is not None and local_idx < len(cache):
            c = cache[local_idx]
        mask = self._mask_for_layer(layer_idx, x, c)
        return self.layers[local_idx](x, mask, c)

    def load_weights(self, weights, strict=False):
        shard_weights: Dict[str, mx.array] = {}
        for key, value in weights:
            if key.startswith("model.layers.") or key.startswith("layers."):
                parts = key.split(".")
                idx_pos = 2 if parts[0] == "model" else 1
                try:
                    abs_idx = int(parts[idx_pos])
                except Exception:
                    continue
                if abs_idx not in self.abs_to_local:
                    continue
                local_idx = self.abs_to_local[abs_idx]
                parts[idx_pos] = str(local_idx)
                if parts[0] == "model":
                    parts = parts[1:]
                new_key = ".".join(parts)
                shard_weights[new_key] = value
            elif self.is_api_layer and (
                key.startswith("embed_tokens") or key.startswith("norm") or key.startswith("lm_head")
            ):
                shard_weights[key] = value
        if shard_weights:
            super().load_weights(list(shard_weights.items()), strict=strict)

    def unload_layers(self, abs_layers: List[int]):
        for abs_idx in abs_layers:
            try:
                local = self.abs_to_local.get(abs_idx)
                if local is None:
                    continue
                block = self.layers[local]
                self._shrink_block(block)
            except Exception:
                continue

    def _shrink_linear_like(self, mod):
        try:
            import mlx.core as _mx
            for name in ("weight", "bias", "scales", "biases"):
                if hasattr(mod, name):
                    arr = getattr(mod, name)
                    try:
                        dt = arr.dtype
                    except Exception:
                        continue
                    if name == "weight":
                        new_arr = _mx.zeros((1, 1), dtype=dt)
                    else:
                        new_arr = _mx.zeros((1,), dtype=dt)
                    setattr(mod, name, new_arr)
        except Exception:
            pass

    def _shrink_block(self, block):
        try:
            if hasattr(block, "self_attn"):
                attn = block.self_attn
                for pn in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    if hasattr(attn, pn):
                        self._shrink_linear_like(getattr(attn, pn))
            if hasattr(block, "mlp"):
                mlp = block.mlp
                for pn in ("router", "gate_proj", "up_proj", "down_proj"):
                    if hasattr(mlp, pn):
                        self._shrink_linear_like(getattr(mlp, pn))
        except Exception:
            pass

    @property
    def decoding_layers(self):
        return self.layers

    @property
    def head_dim(self) -> Tuple[int, int]:
        hd = getattr(self.config, "head_dim", None)
        if hd is None:
            hd = int(self.config.hidden_size // self.config.num_attention_heads)
        return (hd, hd)

    @property
    def n_kv_heads(self) -> int:
        return int(self.config.num_key_value_heads)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def make_cache(self):
        caches: List[Any] = []
        for abs_idx, local_idx in sorted(self.abs_to_local.items()):
            lt = self.layer_types[abs_idx] if 0 <= abs_idx < len(self.layer_types) else "full_attention"
            if lt == "sliding_attention" and self.window_size > 0:
                caches.append(RotatingKVCache(max_size=self.window_size))
            else:
                caches.append(KVCache())
        return caches

