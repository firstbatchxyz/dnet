from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.gpt_oss import ModelArgs, TransformerBlock
from mlx_lm.models.cache import KVCache, RotatingKVCache

from .base import BaseRingModel


class GptOssRingModel(BaseRingModel):
    """Ring-topology, shardable GPT-OSS with MoE + sliding/full attention."""

    model_type = "gpt_oss"

    def __init__(
        self,
        model_config: Any,
        assigned_layers: Optional[List[int]] = None,
        is_api_layer: bool = False,
    ):
        super().__init__()

        if is_api_layer and assigned_layers:
            raise RuntimeError("API layer doesn't handle layers")

        self.model_config = model_config
        self.is_api_layer = is_api_layer
        self.config = cfg = ModelArgs.from_dict(model_config)
        self.layer_types: List[str] = list(
            cfg.layer_types
            or (["sliding_attention", "full_attention"] * (cfg.num_hidden_layers // 2))
        )
        try:
            ws = getattr(cfg, "sliding_window")
        except Exception:
            ws = None
        if ws is None:
            ws = getattr(cfg, "initial_context_length", 128)
        try:
            self.window_size = int(ws)
        except Exception:
            self.window_size = 128

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        self.norm = nn.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if not getattr(cfg, "tie_word_embeddings", False):
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self.layers: List[nn.Module] = []
        self.abs_to_local: Dict[int, int] = {}
        self.local_to_abs: List[int] = []

        for i, abs_idx in enumerate(sorted(assigned_layers or [])):
            self.layers.append(TransformerBlock(cfg))
            self.abs_to_local[abs_idx] = i
            self.local_to_abs.append(abs_idx)

        # Quantization is handled at bind-time in load_weights
        self._converted_to_quantized = False

    def embed(self, x: mx.array) -> mx.array:
        return self.embed_tokens(x) if hasattr(self, "embed_tokens") else x

    def normalize(self, x: mx.array) -> mx.array:
        return self.norm(x) if hasattr(self, "norm") else x

    def lm_project(self, x: mx.array) -> mx.array:
        if hasattr(self, "lm_head") or hasattr(self, "embed_tokens"):
            use_tied = bool(getattr(self.config, "tie_word_embeddings", False))
            if use_tied or not hasattr(self, "lm_head"):
                return self.embed_tokens.as_linear(x)
            return self.lm_head(x)
        return x

    def _pick_cache(
        self, cache: Optional[List[Any]] | Any, abs_idx: int, local_idx: int
    ) -> Optional[Any]:
        """Pick the correct cache object for a given absolute layer.

        Supports both GLOBAL (abs-indexed) cache lists and LOCAL (per-shard) lists.
        If `cache` is not a list, treat it as a single cache object.
        """
        if cache is None:
            return None
        if not isinstance(cache, list):
            return cache
        try:
            total = getattr(self.config, "num_hidden_layers", None)
        except Exception:
            total = None
        # Prefer GLOBAL (abs-indexed) when detectable
        try:
            if (total and len(cache) >= int(total)) and (0 <= abs_idx < len(cache)):
                return cache[abs_idx]
        except Exception:
            pass
        # Otherwise LOCAL (per-shard sized)
        if len(cache) == len(self.layers) and (0 <= local_idx < len(cache)):
            return cache[local_idx]
        # Fallbacks
        if 0 <= abs_idx < len(cache):
            return cache[abs_idx]
        if 0 <= local_idx < len(cache):
            return cache[local_idx]
        return None

    def _build_step_masks(self, x: mx.array, cache: Optional[List[Any]]):
        """Per-step masks: build one full and one sliding mask and reuse within this step.

        Do not cache across steps to ensure KV offsets are respected.
        """
        full_m = None
        swa_m = None
        ga_c = None
        swa_c = None
        ga_abs = None
        swa_abs = None

        # Only build masks we will actually use locally
        local_types = {
            self.layer_types[abs_i]
            if abs_i < len(self.layer_types)
            else "full_attention"
            for abs_i in self.local_to_abs
        }

        # Choose representative local cache for each type present on this shard
        if "full_attention" in local_types:
            rep_li = next(
                (
                    i
                    for i, abs_i in enumerate(self.local_to_abs)
                    if (
                        self.layer_types[abs_i]
                        if abs_i < len(self.layer_types)
                        else "full_attention"
                    )
                    == "full_attention"
                ),
                None,
            )
            if rep_li is not None:
                ga_abs = self.local_to_abs[rep_li]
                ga_c = self._pick_cache(cache, ga_abs, rep_li)
                full_m = create_attention_mask(x, ga_c)

        if "sliding_attention" in local_types:
            rep_li = next(
                (
                    i
                    for i, abs_i in enumerate(self.local_to_abs)
                    if (
                        self.layer_types[abs_i]
                        if abs_i < len(self.layer_types)
                        else "full_attention"
                    )
                    == "sliding_attention"
                ),
                None,
            )
            if rep_li is not None:
                swa_abs = self.local_to_abs[rep_li]
                swa_c = self._pick_cache(cache, swa_abs, rep_li)
                swa_m = create_attention_mask(x, swa_c, window_size=self.window_size)

        return full_m, swa_m, ga_c, swa_c, ga_abs, swa_abs

    def forward(self, x: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        if cache is None:
            cache = [None] * len(self.layers)

        full_m, swa_m, _, _, _, _ = self._build_step_masks(x, cache)

        for i, layer in enumerate(self.layers):
            abs_idx = self.local_to_abs[i]
            lt = (
                self.layer_types[abs_idx]
                if abs_idx < len(self.layer_types)
                else "full_attention"
            )
            c = self._pick_cache(cache, abs_idx, i)
            m = full_m if lt == "full_attention" else swa_m
            # If mask for this type was not built (should not happen if local_types contains it), build on-demand
            if m is None:
                if lt == "full_attention":
                    m = create_attention_mask(x, c)
                else:
                    m = create_attention_mask(x, c, window_size=self.window_size)
            x = layer(x, m, c)

        return x

    def apply_single_layer(
        self, layer_idx: int, x: mx.array, cache: Optional[List[Any]] = None
    ) -> mx.array:
        if layer_idx not in self.abs_to_local:
            raise RuntimeError(f"Layer {layer_idx} not hosted on this model instance")

        local_idx = self.abs_to_local[layer_idx]
        c = self._pick_cache(cache, layer_idx, local_idx)
        lt = (
            self.layer_types[layer_idx]
            if layer_idx < len(self.layer_types)
            else "full_attention"
        )
        if lt == "full_attention":
            mask = create_attention_mask(x, c)
        else:
            mask = create_attention_mask(x, c, window_size=self.window_size)
        return self.layers[local_idx](x, mask, c)

    def sanitize_weights(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Tensor-level normalization
        """
        if any("gate_proj.weight" in k for k in weights.keys()):
            return weights

        new_w = {}
        for k, v in weights.items():
            if "gate_up_proj" in k and "bias" not in k:
                if "_blocks" in k:
                    v = v.view(mx.uint32).flatten(-2)
                    k = k.replace("_blocks", ".weight")
                if "_scales" in k:
                    k = k.replace("_scales", ".scales")
                new_w[k.replace("gate_up_proj", "gate_proj")] = mx.contiguous(
                    v[..., ::2, :]
                )
                new_w[k.replace("gate_up_proj", "up_proj")] = mx.contiguous(
                    v[..., 1::2, :]
                )
            elif "down_proj" in k and "bias" not in k:
                if "_blocks" in k:
                    v = v.view(mx.uint32).flatten(-2)
                    k = k.replace("_blocks", ".weight")
                if "_scales" in k:
                    k = k.replace("_scales", ".scales")
                new_w[k] = v
            elif "gate_up_proj_bias" in k:
                new_w[k.replace("gate_up_proj_bias", "gate_proj.bias")] = mx.contiguous(
                    v[..., ::2]
                )
                new_w[k.replace("gate_up_proj_bias", "up_proj.bias")] = mx.contiguous(
                    v[..., 1::2]
                )
            elif "down_proj_bias" in k:
                new_w[k.replace("down_proj_bias", "down_proj.bias")] = v
            else:
                new_w[k] = v
        return new_w

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Config-level normalization
        """
        if any("gate_proj.weight" in k for k in weights.keys()):
            return weights
        new_w: Dict[str, Any] = {}
        for k, v in weights.items():
            if "gate_up_proj" in k and "bias" not in k:
                k_eff = k
                if "_blocks" in k_eff:
                    k_eff = k_eff.replace("_blocks", ".weight")
                if "_scales" in k_eff:
                    k_eff = k_eff.replace("_scales", ".scales")
                new_w[k_eff.replace("gate_up_proj", "gate_proj")] = v
                new_w[k_eff.replace("gate_up_proj", "up_proj")] = v
            elif "down_proj" in k and "bias" not in k:
                k_eff = k
                if "_blocks" in k_eff:
                    k_eff = k_eff.replace("_blocks", ".weight")
                if "_scales" in k_eff:
                    k_eff = k_eff.replace("_scales", ".scales")
                new_w[k_eff] = v
            elif "gate_up_proj_bias" in k:
                new_w[k.replace("gate_up_proj_bias", "gate_proj.bias")] = v
                new_w[k.replace("gate_up_proj_bias", "up_proj.bias")] = v
            elif "down_proj_bias" in k:
                new_w[k.replace("down_proj_bias", "down_proj.bias")] = v
            else:
                new_w[k] = v
        return new_w

    # load_weights inherited from BaseRingModel

    def make_cache(self) -> List[Any]:
        caches: List[Any] = []
        for abs_idx in self.local_to_abs:
            lt = (
                self.layer_types[abs_idx]
                if abs_idx < len(self.layer_types)
                else "full_attention"
            )
            if lt == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.window_size))
        return caches

    @property
    def decoding_layers(self):
        return self.layers

    @property
    def head_dim(self) -> Tuple[int, int]:
        return (self.config.head_dim, self.config.head_dim)

    @property
    def n_kv_heads(self) -> int:
        return self.config.num_key_value_heads

    @property
    def num_layers(self) -> int:
        return len(self.layers)
