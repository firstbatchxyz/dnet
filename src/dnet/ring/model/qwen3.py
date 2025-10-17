from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3 import ModelArgs, TransformerBlock

from .base import BaseRingModel
from ...utils.logger import logger


class Qwen3RingModel(BaseRingModel):
    """Qwen3 model for ring topology inference.
    Supports layer-wise application and distributed inference.
    """

    model_type = "qwen3"

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
        self.config = config = ModelArgs.from_dict(model_config)

        logger.info(
            "Initializing Qwen3RingModel: is_api_layer=%s, assigned_layers=%s",
            is_api_layer,
            assigned_layers,
        )
        logger.info(
            "Config: hidden_size=%d, num_heads=%d, num_kv_heads=%d",
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
        )

        # Create embed, norm, head
        if "quantization" in model_config:
            # Use quantized embedding module to match on-disk packed tensors
            try:
                from mlx.nn.layers.quantized import QuantizedEmbedding

                qcfg = model_config["quantization"]
                bits = int(qcfg.get("bits", 8))
                group = int(qcfg.get("group_size", 64))
                self.embed_tokens = QuantizedEmbedding(
                    config.vocab_size, config.hidden_size, group_size=group, bits=bits
                )
                logger.info(
                    "embed_tokens -> QuantizedEmbedding: vocab=%d hidden=%d group_size=%d bits=%d",
                    config.vocab_size,
                    config.hidden_size,
                    group,
                    bits,
                )
            except Exception as _e:
                logger.warning(
                    "QuantizedEmbedding unavailable (%s); using dense Embedding", _e
                )
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Qwen3 can tie embeddings; add head only if not tied
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Other layers: local zero-based list and abs->local map
        self.layers: List[nn.Module] = []
        self.abs_to_local: Dict[int, int] = {}

        # Check if model is quantized from config
        self.is_quantized = "quantization" in model_config
        if self.is_quantized:
            self.quantization_config = model_config["quantization"]
            logger.info("Model is quantized with config: %s", self.quantization_config)

        # Create TransformerBlocks for assigned layers
        for i, layer in enumerate(sorted(assigned_layers or [])):
            self.layers.append(TransformerBlock(config))
            self.abs_to_local[layer] = i

        if self.is_quantized and (not is_api_layer):
            try:
                # Disable lazy shrinking for quantized to avoid (1,1) placeholders
                # (also enforced below before any optional shrinking).
                bits = (
                    int(self.quantization_config.get("bits", 8))
                    if hasattr(self, "quantization_config")
                    else 8
                )
                group = (
                    int(self.quantization_config.get("group_size", 64))
                    if hasattr(self, "quantization_config")
                    else 64
                )

                # Quantize all nn.Linear modules in the decoder blocks
                def _quant_pred(p, m):
                    return isinstance(m, nn.Linear)

                nn.quantize(
                    self, bits=bits, group_size=group, class_predicate=_quant_pred
                )
                self._converted_to_quantized = True
                logger.debug(
                    "Applied init-time quantization: bits=%d group_size=%d", bits, group
                )
            except Exception as _e:
                logger.warning("Init-time quantization failed: %s", _e)
        elif not is_api_layer:
            logger.debug("Skipping init-time quantization; will keep Linear layers")

        logger.debug("Created %d TransformerBlock layers", len(self.layers))
        logger.debug("abs_to_local mapping: %s", self.abs_to_local)

        self._converted_to_quantized = False
        # Allow API to force using tied embeddings as final head if standalone
        # lm_head weights are unavailable or incompatible
        self.force_tied_head: bool = False

        # When enabled, we replace all per-layer params with tiny placeholders
        # at construction time; real arrays are bound only when weights are
        # loaded for the active window. On eviction, we shrink them again.
        import os as _os

        try:
            self._lazy_params = _os.getenv("RING_LAZY_PARAMS", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        except Exception:
            self._lazy_params = False
        # Do not shrink params for quantized models; conversion relies on real shapes
        if self.is_quantized and self._lazy_params:
            logger.info(
                "Ignoring RING_LAZY_PARAMS for quantized model to preserve parameter shapes"
            )
            self._lazy_params = False
        if (not self.is_api_layer) and self._lazy_params:
            try:
                self._shrink_all_params()
                logger.info(
                    "Enabled lazy params: per-layer arrays are placeholders until bound by weights"
                )
            except Exception as _e:
                logger.warning(f"Lazy-param shrink failed: {_e}")

        # Compiled window cache: absolute-layer tuple -> compiled forward fn
        # not used yet
        self._compiled_windows: dict[tuple[int, ...], Any] = {}
        # Runtime KVCache
        self._runtime_cache: Optional[List[Any]] = None

    def set_runtime_cache(self, cache: Optional[List[Any]]) -> None:
        """Set the runtime KV cache reference for compiled decode windows."""
        self._runtime_cache = cache

    @staticmethod
    def class_predicate(p, m):
        return hasattr(m, "to_quantized")

    def apply_quantization(self):
        """Apply quantization after weights are loaded"""
        # Skip if this is the API layer
        if self.is_api_layer:
            return

        # Check if model is already quantized by checking if any Linear layers are QuantizedLinear
        from mlx.nn.layers.quantized import QuantizedLinear

        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, QuantizedLinear):
                    # Model is already quantized, skip re-quantization
                    return

        # Only quantize if not already quantized and quantization config exists
        if "quantization" in self.model_config:
            quant_config = self.model_config["quantization"].copy()
            nn.quantize(
                self,
                **quant_config,
                class_predicate=Qwen3RingModel.class_predicate,
            )

    def _shrink_linear_like(self, mod):
        try:
            import mlx.core as _mx

            for name in ("weight", "bias", "scales", "biases"):
                if hasattr(mod, name):
                    arr = getattr(mod, name)
                    try:
                        # shp = tuple(arr.shape)
                        dt = arr.dtype
                    except Exception:
                        continue
                    # Minimal shapes for placeholders
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
            # Attention projections
            if hasattr(block, "self_attn"):
                attn = block.self_attn
                for pn in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    if hasattr(attn, pn):
                        self._shrink_linear_like(getattr(attn, pn))
            # MLP projections
            if hasattr(block, "mlp"):
                mlp = block.mlp
                for pn in ("gate_proj", "up_proj", "down_proj"):
                    if hasattr(mlp, pn):
                        self._shrink_linear_like(getattr(mlp, pn))
        except Exception:
            pass

    def _shrink_all_params(self):
        for b in self.layers:
            self._shrink_block(b)

    def unload_layers(self, abs_layers: list[int]):
        """Shrink params for given absolute layer ids to placeholders.

        Call after a window is evicted to free memory retained by module params.
        """
        # Do not shrink quantized models to avoid introducing (1,1) placeholders
        # in QuantizedLinear modules which will break subsequent matmuls.
        if self.is_quantized:
            return
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
        if hasattr(self, "embed_tokens"):
            return self.embed_tokens(x)
        return x

    def normalize(self, x: mx.array) -> mx.array:
        if hasattr(self, "norm"):
            return self.norm(x)
        return x

    def lm_project(self, x: mx.array) -> mx.array:
        if hasattr(self, "lm_head") or hasattr(self, "embed_tokens"):
            use_tied = bool(getattr(self, "force_tied_head", False)) or bool(
                getattr(self.config, "tie_word_embeddings", False)
            )
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
        mask = None
        if T > 1:
            cached = getattr(self, "_cached_mask_state", None)
            if cached is None or cached != T:
                mask = create_attention_mask(x, cache)
                self._cached_mask = mask
                self._cached_mask_state = T
            else:
                mask = getattr(self, "_cached_mask", None)
                if mask is None:
                    mask = create_attention_mask(x, cache)
                    self._cached_mask = mask
                    self._cached_mask_state = T
        local_idx = self.abs_to_local[layer_idx]

        logger.debug(
            "apply_single_layer: layer_idx=%d, local_idx=%d, input shape=%s",
            layer_idx,
            local_idx,
            x.shape,
        )

        # Log the layer's weight shapes
        layer = self.layers[local_idx]
        if hasattr(layer, "self_attn"):
            if hasattr(layer.self_attn, "q_proj"):
                if hasattr(layer.self_attn.q_proj, "weight"):
                    logger.debug(
                        "Layer %d q_proj weight shape: %s",
                        layer_idx,
                        layer.self_attn.q_proj.weight.shape,
                    )
                else:
                    logger.debug("Layer %d q_proj has no weight attribute", layer_idx)
            if hasattr(layer.self_attn, "k_proj"):
                if hasattr(layer.self_attn.k_proj, "weight"):
                    logger.debug(
                        "Layer %d k_proj weight shape: %s",
                        layer_idx,
                        layer.self_attn.k_proj.weight.shape,
                    )
                else:
                    logger.debug("Layer %d k_proj has no weight attribute", layer_idx)

        c = None
        if cache is not None and local_idx < len(cache):
            c = cache[local_idx]

        result = self.layers[local_idx](x, mask, c)
        logger.debug("Layer %d output shape: %s", layer_idx, result.shape)
        return result

    def load_weights(self, weights, strict=False):
        """Load weights into the model"""
        logger.info("load_weights called with %s weights", len(weights))
        logger.debug("First few weight keys: %s", [k for k, _ in weights[:5]])
        logger.debug("abs_to_local mapping: %s", self.abs_to_local)

        # Filter weights to only include what this shard needs
        shard_weights = {}

        for key, value in weights:
            # Accept both bare 'layers.*' and 'model.layers.*' and remap abs->local
            if key.startswith("model.layers.") or key.startswith("layers."):
                parts = key.split(".")
                idx_pos = 2 if parts[0] == "model" else 1
                try:
                    abs_idx = int(parts[idx_pos])
                except Exception:
                    continue
                if abs_idx not in self.abs_to_local:
                    # Skip layers not assigned to this shard
                    continue
                local_idx = self.abs_to_local[abs_idx]
                # Keep the "layers" prefix and remap index
                parts[idx_pos] = str(local_idx)
                # Remove "model." prefix if present
                if parts[0] == "model":
                    parts = parts[1:]
                new_key = ".".join(parts)
                logger.debug(
                    "Mapping weight %s (shape %s) -> %s", key, value.shape, new_key
                )
                shard_weights[new_key] = value

            elif key.startswith("embed_tokens"):
                shard_weights[key] = value

                logger.info("API layer: loading %s, shape=%s", key, value.shape)
            elif key.startswith("norm"):
                shard_weights[key] = value
                logger.info("API layer: loading %s, shape=%s", key, value.shape)

            elif key.startswith("lm_head") and not self.config.tie_word_embeddings:
                shard_weights[key] = value
                logger.info("API layer: loading %s, shape=%s", key, value.shape)

        logger.info("Loading %d weights into model", len(shard_weights))

        if shard_weights:
            # Log the first weight being loaded to check dimensions
            first_key = list(shard_weights.keys())[0]
            logger.info(
                "First weight to load: %s with shape %s",
                first_key,
                shard_weights[first_key].shape,
            )

            if "layers." in first_key:
                layer_idx = first_key.split(".")[1]
                logger.info("Loading into local layer %s", layer_idx)
                logger.info("Number of layers in model: %d", len(self.layers))
                if int(layer_idx) < len(self.layers):
                    layer = self.layers[int(layer_idx)]
                    # Log the current layer structure
                    if hasattr(layer, "self_attn"):
                        if hasattr(layer.self_attn, "q_proj"):
                            logger.info(
                                "Current q_proj type: %s", type(layer.self_attn.q_proj)
                            )
                            if hasattr(layer.self_attn.q_proj, "weight"):
                                logger.info(
                                    "Current q_proj weight shape: %s",
                                    layer.self_attn.q_proj.weight.shape,
                                )

        # Load the filtered weights using parent class method
        try:
            super().load_weights(list(shard_weights.items()), strict=strict)
            logger.info("Successfully loaded weights")

            # Skip numeric stats for quantized modules; values are not meaningful
            if (not self.is_quantized) and shard_weights and "layers." in first_key:
                layer_idx = first_key.split(".")[1]
                if int(layer_idx) < len(self.layers):
                    layer = self.layers[int(layer_idx)]
                    if hasattr(layer, "self_attn") and hasattr(
                        layer.self_attn, "q_proj"
                    ):
                        if hasattr(layer.self_attn.q_proj, "weight"):
                            weight = layer.self_attn.q_proj.weight
                            logger.info(
                                "After loading - q_proj weight stats: shape=%s, mean=%.6f, std=%.6f",
                                weight.shape,
                                mx.mean(weight).item(),
                                mx.std(weight).item(),
                            )
                            if (
                                mx.abs(mx.mean(weight)).item() < 1e-6
                                and mx.std(weight).item() < 1e-6
                            ):
                                logger.warning(
                                    "WARNING: q_proj weights appear to be all zeros!"
                                )
                            elif mx.std(weight).item() > 1.0:
                                logger.warning(
                                    "WARNING: q_proj weights have very high std dev, might be uninitialized!"
                                )
        except Exception as e:
            logger.error("Failed to load weights: %s", e)
            logger.error("Weight keys: %s", list(shard_weights.keys()))
            raise

    @property
    def decoding_layers(self):
        return self.layers

    @property
    def head_dim(self) -> Tuple[int, int]:
        # Qwen3 uses the same head_dim for both Q and V
        return (self.config.head_dim, self.config.head_dim)

    @property
    def n_kv_heads(self) -> int:
        return self.config.num_key_value_heads

    @property
    def num_layers(self) -> int:
        # Number of local decoding layers hosted on this shard model
        return len(self.layers)
