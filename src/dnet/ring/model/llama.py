from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.llama import ModelArgs, TransformerBlock

from .base import BaseRingModel
from ...utils.logger import logger


class LlamaRingModel(BaseRingModel):
    """Llama model for ring topology inference.

    Mirrors mlx-lm's Llama modules but constructs only the locally assigned
    decoder blocks and exposes layer-wise application.
    """

    model_type = "llama"

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

        self.model_config = model_config
        self.is_api_layer = is_api_layer
        self.config = config = ModelArgs.from_dict(model_config)

        if "quantization" in model_config:
            try:
                from mlx.nn.layers.quantized import QuantizedEmbedding  # type: ignore

                qcfg = model_config["quantization"]
                bits = int(qcfg.get("bits", 8))
                group = int(qcfg.get("group_size", 64))
                self.embed_tokens = QuantizedEmbedding(
                    config.vocab_size, config.hidden_size, group_size=group, bits=bits
                )
            except Exception as _e:
                logger.warning(
                    "QuantizedEmbedding unavailable (%s); using dense Embedding", _e
                )
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.layers: List[nn.Module] = []
        self.abs_to_local: Dict[int, int] = {}

        self.is_quantized = "quantization" in model_config
        if self.is_quantized:
            self.quantization_config = model_config["quantization"]

        for i, layer in enumerate(sorted(assigned_layers or [])):
            self.layers.append(TransformerBlock(config))
            self.abs_to_local[layer] = i

        # Optional post-init quantization of Linear layers for offload shards
        if self.is_quantized and (not is_api_layer):
            try:
                bits = int(self.quantization_config.get("bits", 8))
                group = int(self.quantization_config.get("group_size", 64))

                def _quant_pred(p, m):
                    return isinstance(m, nn.Linear)

                nn.quantize(self, bits=bits, group_size=group, class_predicate=_quant_pred)
                self._converted_to_quantized = True
            except Exception as _e:
                logger.warning("Init-time quantization failed: %s", _e)
        else:
            self._converted_to_quantized = False

        self.force_tied_head: bool = False
        self._cached_mask_state = None
        self._cached_mask = None

        # Init-time lazy param shrink removed to simplify startup behavior.

    def _shrink_linear_like(self, mod):
        try:
            import mlx.core as _mx

            for name in ("weight", "scales", "biases", "bias"):
                if hasattr(mod, name):
                    arr = getattr(mod, name)
                    try:
                        dt = arr.dtype
                    except Exception:
                        dt = _mx.float16
                    if name == "weight":
                        new_arr = _mx.zeros((1, 1), dtype=dt)
                    elif name in ("scales", "biases", "bias"):
                        new_arr = _mx.zeros((1,), dtype=dt)
                    else:
                        continue
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
                for pn in ("gate_proj", "up_proj", "down_proj"):
                    if hasattr(mlp, pn):
                        self._shrink_linear_like(getattr(mlp, pn))
        except Exception:
            pass

    def unload_layers(self, abs_layers: List[int]):
        # Shrink params for given absolute layer ids to placeholders.
        # Safe for both quantized and unquantized models under windowed execution;
        # layers will be re-bound before next use.
        for abs_idx in abs_layers:
            try:
                local = self.abs_to_local.get(abs_idx)
                if local is None:
                    continue
                block = self.layers[local]
                self._shrink_block(block)
            except Exception:
                continue

    def embed(self, x: mx.array) -> mx.array:
        return self.embed_tokens(x)

    def normalize(self, x: mx.array) -> mx.array:
        return self.norm(x)

    def lm_project(self, x: mx.array) -> mx.array:
        use_tied = bool(self.force_tied_head) or bool(getattr(self.config, "tie_word_embeddings", False))
        if use_tied or not hasattr(self, "lm_head"):
            return self.embed_tokens.as_linear(x)
        return self.lm_head(x)

    def forward(self, x: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        mask = create_attention_mask(x, cache)
        if cache is None:
            cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, cache[i] if i < len(cache) else None)
        return x

    def apply_single_layer(
        self, layer_idx: int, x: mx.array, cache: Optional[List[Any]] = None
    ) -> mx.array:
        if layer_idx not in self.abs_to_local:
            raise RuntimeError(f"Layer {layer_idx} not hosted on this model instance")
        # Create/reuse attention mask when T > 1
        try:
            T = int(x.shape[1]) if len(x.shape) > 1 else 1
        except Exception:
            T = 1
        mask = None
        if T > 1:
            if self._cached_mask_state is None or self._cached_mask_state != T:
                mask = create_attention_mask(x, cache)
                self._cached_mask = mask
                self._cached_mask_state = T
            else:
                mask = self._cached_mask
                if mask is None:
                    mask = create_attention_mask(x, cache)
                    self._cached_mask = mask
                    self._cached_mask_state = T
        local_idx = self.abs_to_local[layer_idx]
        c = None
        if cache is not None and local_idx < len(cache):
            c = cache[local_idx]
        return self.layers[local_idx](x, mask, c)

    def load_weights(self, weights, strict=False):
        """Load only local layer and API tensors. Remap abs->local indices.
        Accepts keys in either 'model.layers.N.*' or 'layers.N.*' formats.
        """
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
            elif key.startswith("embed_tokens") or key.startswith("norm") or (
                key.startswith("lm_head") and not self.config.tie_word_embeddings
            ):
                shard_weights[key] = value

        try:
            super().load_weights(list(shard_weights.items()), strict=strict)
        except Exception as e:
            logger.error("Failed to load weights: %s", e)
            logger.error("Weight keys: %s", list(shard_weights.keys()))
            raise

    @property
    def decoding_layers(self):
        return self.layers

    @property
    def head_dim(self) -> Tuple[int, int]:
        head_dim = self.config.head_dim or self.config.hidden_size // self.config.num_attention_heads
        return (head_dim, head_dim)

    @property
    def n_kv_heads(self) -> int:
        return self.config.num_key_value_heads

    @property
    def num_layers(self) -> int:
        return len(self.layers)
