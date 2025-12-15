from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3_moe import ModelArgs, Attention, MLP, SwitchGLU

from .base import BaseRingModel


class Qwen3MoEDenseBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_experts = config.num_experts
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, self.num_experts
        )

    def __call__(self, x: mx.array):
        indices = mx.arange(self.num_experts, dtype=mx.int32)
        indices = mx.broadcast_to(
            indices[None, None, :], (x.shape[0], x.shape[1], self.num_experts)
        )
        gates = self.gate(x)
        scores = mx.softmax(gates, axis=-1, precise=True)
        y = self.switch_mlp(x, indices)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


# force dense execution
class Qwen3MoEDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = Attention(config, layer_idx)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoEDenseBlock(config)
        else:
            self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3MoERingModel(BaseRingModel):
    """Qwen3 MoE model for distributed execution"""

    model_type = "qwen3_moe"

    def __init__(
        self,
        model_config: Any,
        assigned_layers: Optional[List[int]] = None,
        is_api_layer: bool = False,
    ):
        super().__init__()

        if is_api_layer and assigned_layers:
            raise RuntimeError("API Node cannot execute layers")

        self.model_config = model_config
        self.is_api_layer = is_api_layer
        self.config = config = ModelArgs.from_dict(model_config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.layers: List[nn.Module] = []
        self.abs_to_local: Dict[int, int] = {}

        for i, layer in enumerate(sorted(assigned_layers or [])):
            self.layers.append(Qwen3MoEDecoderLayer(config, layer_idx=layer))
            self.abs_to_local[layer] = i

        self._converted_to_quantized = False
        self._cached_mask_state: Optional[int] = None
        self._cached_mask = None

    def embed(self, x: mx.array) -> mx.array:
        if hasattr(self, "embed_tokens"):
            return self.embed_tokens(x)
        return x

    def normalize(self, x: mx.array) -> mx.array:
        if hasattr(self, "norm"):
            return self.norm(x)
        return x

    def lm_project(self, x: mx.array) -> mx.array:
        if hasattr(self, "lm_head") or hasattr(self, "embed_tokens"):
            use_tied = bool(getattr(self.config, "tie_word_embeddings", False))
            if use_tied or not hasattr(self, "lm_head"):
                return self.embed_tokens.as_linear(x)
            return self.lm_head(x)
        return x

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
        #  TODO: Mask reuse should respect concurrent requests
        try:
            T = int(x.shape[1]) if len(x.shape) > 1 else 1
        except Exception:
            T = 1
        # dimension diagnostics removed
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

    # qwen stores expert weights in separate buffers
    def sanitize(self, weights):
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights
        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                if f"{prefix}.mlp.experts.0.{n}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                        for e in range(self.config.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)
        return weights

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
