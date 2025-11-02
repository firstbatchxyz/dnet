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

        q_overrides = model_config.get("quantization", {}) or {}
        try:
            if any(k.startswith("model.embed_tokens") for k in q_overrides.keys()):
                from mlx.nn.layers.quantized import QuantizedEmbedding  # type: ignore
                ov = next(
                    (v for k, v in q_overrides.items() if k.startswith("model.embed_tokens")),
                    {"bits": 8, "group_size": 64},
                )
                bits = int(ov.get("bits", 8))
                group = int(ov.get("group_size", 64))
                self.embed_tokens = QuantizedEmbedding(
                    config.vocab_size, config.hidden_size, group_size=group, bits=bits
                )
            else:
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        except Exception:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if not getattr(config, "tie_word_embeddings", False):
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )

        self.layers: List[nn.Module] = []
        self.abs_to_local: Dict[int, int] = {}
        for i, layer in enumerate(sorted(assigned_layers or [])):
            self.layers.append(TransformerBlock(config))
            self.abs_to_local[layer] = i

        self._cached_mask_state: Optional[int] = None
        self._cached_full_mask = None
        self._cached_swa_mask = None

        self._converted_to_quantized = False
        try:
            if not is_api_layer:
                qc = model_config.get("quantization_config", {}) or {}
                default_method = str(qc.get("quant_method", "mxfp4")).strip().lower()
                modules_to_not_convert = list(qc.get("modules_to_not_convert", []) or [])

                overrides_src = model_config.get("quantization", {}) or {}
                override_groups: dict[tuple[int, int, str], set[str]] = {}
                for k, v in overrides_src.items():
                    try:
                        bits = int(v.get("bits", 8))
                    except Exception:
                        bits = 8
                    try:
                        group = int(v.get("group_size", 64))
                    except Exception:
                        group = 64
                    mode = str(v.get("mode", "affine")).strip().lower()
                    # Map model.* absolute key to ring-local module path
                    path = self._abskey_to_local_path(k)
                    if path is None:
                        continue
                    override_groups.setdefault((bits, group, mode), set()).add(path)

                import fnmatch

                def _excluded(path: str) -> bool:
                    for pat in modules_to_not_convert:
                        pat_norm = pat.replace("model.", "")
                        if fnmatch.fnmatch(path, pat_norm) or path.endswith(pat_norm):
                            return True
                    return False

                for (bits, group, mode), paths in override_groups.items():
                    def _pred_override(p, m, paths=paths):
                        return isinstance(m, nn.Linear) and (p in paths)
                    try:
                        nn.quantize(
                            self,
                            bits=bits,
                            group_size=group,
                            class_predicate=_pred_override,
                            mode=mode,  # type: ignore[call-arg]
                        )
                    except TypeError:
                        nn.quantize(
                            self, bits=bits, group_size=group, class_predicate=_pred_override
                        )
                    self._converted_to_quantized = True

                if default_method:
                    def _pred_default(p, m):
                        if not isinstance(m, nn.Linear):
                            return False
                        if _excluded(p):
                            return False
                        # Skip modules covered by overrides
                        for _, paths in override_groups.items():
                            if p in paths:
                                return False
                        return True

                    try:
                        nn.quantize(
                            self,
                            bits=4,
                            group_size=32,
                            class_predicate=_pred_default,
                            mode=default_method,  # type: ignore[call-arg]
                        )
                    except TypeError:
                        nn.quantize(
                            self,
                            bits=4,
                            group_size=32,
                            class_predicate=_pred_default,
                        )
                    self._converted_to_quantized = True
        except Exception:
            self._converted_to_quantized = False

    def _abskey_to_local_path(self, key: str) -> Optional[str]:
        # Map 'model.layers.ABS.suffix' → 'layers.LOCAL.suffix'
        # Also pass through top-level 'model.embed_tokens' and 'lm_head'
        try:
            if key.startswith("model.layers."):
                parts = key.split(".")
                abs_idx = int(parts[2])
                local = self.abs_to_local.get(abs_idx)
                if local is None:
                    return None
                suffix = ".".join(parts[3:])
                return f"layers.{local}.{suffix}"
            if key.startswith("model.embed_tokens"):
                return "embed_tokens"
            if key.startswith("lm_head"):
                return "lm_head"
            return None
        except Exception:
            return None

    def embed(self, x: mx.array) -> mx.array:
        return self.embed_tokens(x) if self.is_api_layer else x

    def normalize(self, x: mx.array) -> mx.array:
        return self.norm(x) if self.is_api_layer else x

    def lm_project(self, x: mx.array) -> mx.array:
        use_tied = bool(getattr(self.config, "tie_word_embeddings", False))
        if use_tied or not hasattr(self, "lm_head"):
            return self.embed_tokens.as_linear(x)
        return self.lm_head(x)

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
            elif key.startswith("embed_tokens") or key.startswith("norm") or key.startswith("lm_head"):
                shard_weights[key] = value

        if shard_weights:
            shard_weights = self._sanitize_weights(shard_weights)
            super().load_weights(list(shard_weights.items()), strict=strict)

    def _sanitize_weights(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # Mirror mlx-lm GPT-OSS sanitize behavior for fused gate_up_proj and quant keys
        if any("gate_proj.weight" in k for k in weights.keys()):
            return weights
        new_weights: Dict[str, mx.array] = {}
        for k, v in weights.items():
            if "gate_up_proj" in k and "bias" not in k:
                if "_blocks" in k:
                    v = v.view(mx.uint32).flatten(-2)
                    k = k.replace("_blocks", ".weight")
                if "_scales" in k:
                    k = k.replace("_scales", ".scales")
                new_weights[k.replace("gate_up_proj", "gate_proj")] = mx.contiguous(
                    v[..., ::2, :]
                )
                new_weights[k.replace("gate_up_proj", "up_proj")] = mx.contiguous(
                    v[..., 1::2, :]
                )
            elif "down_proj" in k and "bias" not in k:
                if "_blocks" in k:
                    v = v.view(mx.uint32).flatten(-2)
                    k = k.replace("_blocks", ".weight")
                if "_scales" in k:
                    k = k.replace("_scales", ".scales")
                new_weights[k] = v
            elif "gate_up_proj_bias" in k:
                new_weights[k.replace("gate_up_proj_bias", "gate_proj.bias")] = (
                    mx.contiguous(v[..., ::2])
                )
                new_weights[k.replace("gate_up_proj_bias", "up_proj.bias")] = (
                    mx.contiguous(v[..., 1::2])
                )
            elif "down_proj_bias" in k:
                new_weights[k.replace("down_proj_bias", "down_proj.bias")] = v
            else:
                new_weights[k] = v
        return new_weights

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
            for name in ("weight", "bias", "scales", "biases"):
                if hasattr(mod, name):
                    arr = getattr(mod, name)
                    try:
                        dt = arr.dtype
                    except Exception:
                        continue
                    if name == "weight":
                        new_arr = mx.zeros((1, 1), dtype=dt)
                    else:
                        new_arr = mx.zeros((1,), dtype=dt)
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
