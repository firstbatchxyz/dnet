

from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3 import ModelArgs, MLP 
from src.runtime.sparse_attention import SparseAttention, FlexPrefillSparseAttention

# Sparse attention transformer block
class TransformerBlock(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.hidden_size = args.hidden_size
    self.num_attention_heads = args.num_attention_heads
    self.mlp = MLP(args.hidden_size, args.intermediate_size)
    self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    self.self_attn = SparseAttention(args, FlexPrefillSparseAttention())

  def __call__(
    self, 
    x:mx.array, 
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
  ) -> mx.array:
    r = self.self_attn(self.input_layernorm(x), mask, cache)
    h = x + r
    r = self.mlp(self.post_attention_layernorm(x))
    return h + r


from .base import BaseRingModel
import logging

logger = logging.getLogger(__name__)


class Qwen3RingModel(BaseRingModel):
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
            f"Initializing Qwen3RingModel: is_api_layer={is_api_layer}, assigned_layers={assigned_layers}"
        )
        logger.info(
            f"Config: hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, num_kv_heads={config.num_key_value_heads}"
        )

        # The API layer handles embedding and normalization
        if is_api_layer:
            # Start with regular Embedding, will be converted if needed when loading quantized weights
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            # Qwen3 can tie embeddings; add head only if not tied
            if not config.tie_word_embeddings:
                self.lm_head = nn.Linear(
                    config.hidden_size, config.vocab_size, bias=False
                )

        # Other layers: local zero-based list and abs->local map
        self.layers: List[nn.Module] = []
        self.abs_to_local: Dict[int, int] = {}

        # Check if model is quantized from config
        self.is_quantized = "quantization" in model_config
        if self.is_quantized:
            self.quantization_config = model_config["quantization"]
            logger.info(f"Model is quantized with config: {self.quantization_config}")

        # For now, create regular TransformerBlocks
        # They will be converted to quantized on first weight load if needed
        for i, layer in enumerate(sorted(assigned_layers or [])):
            self.layers.append(TransformerBlock(config))
            self.abs_to_local[layer] = i

        # For shard layers (non-API), enable quantization upfront if configured.
        # This ensures QuantizedLinear modules are in place before loading
        # per-layer int weights/scales from the weight cache.
        if not is_api_layer and self.is_quantized:
            logger.info("Applying quantization for shard layers")
            self.apply_quantization()
        elif not is_api_layer:
            logger.info("Not applying quantization - model is not quantized")

        logger.info(f"Created {len(self.layers)} sparse TransformerBlocks layers")
        #logger.info(f"Created {len(self.layers)} TransformerBlock layers")
        logger.info(f"abs_to_local mapping: {self.abs_to_local}")

        # Flag to track if layers have been converted to quantized
        self._converted_to_quantized = False

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

    def embed(self, x: mx.array) -> mx.array:
        return self.embed_tokens(x) if self.is_api_layer else x

    def normalize(self, x: mx.array) -> mx.array:
        return self.norm(x) if self.is_api_layer else x

    def lm_project(self, x: mx.array) -> mx.array:
        if self.is_api_layer:
            if self.config.tie_word_embeddings:
                # For tied embeddings, use embed_tokens as linear projection
                return self.embed_tokens.as_linear(x)
            return self.lm_head(x)
        return x

    def forward(self, x: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        # Create attention mask
        mask = create_attention_mask(x, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        # Apply in local order 0..len-1
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, cache[i] if i < len(cache) else None)

        return x

    def apply_single_layer(
        self, layer_idx: int, x: mx.array, cache: Optional[List[Any]] = None
    ) -> mx.array:
        if layer_idx not in self.abs_to_local:
            raise RuntimeError(f"Layer {layer_idx} not hosted on this model instance")

        mask = create_attention_mask(x, cache)
        local_idx = self.abs_to_local[layer_idx]

        logger.info(
            f"apply_single_layer: layer_idx={layer_idx}, local_idx={local_idx}, input shape={x.shape}"
        )

        # Log the layer's weight shapes
        layer = self.layers[local_idx]
        if hasattr(layer, "self_attn"):
            if hasattr(layer.self_attn, "q_proj"):
                if hasattr(layer.self_attn.q_proj, "weight"):
                    logger.info(
                        f"Layer {layer_idx} q_proj weight shape: {layer.self_attn.q_proj.weight.shape}"
                    )
                else:
                    logger.info(f"Layer {layer_idx} q_proj has no weight attribute")
            if hasattr(layer.self_attn, "k_proj"):
                if hasattr(layer.self_attn.k_proj, "weight"):
                    logger.info(
                        f"Layer {layer_idx} k_proj weight shape: {layer.self_attn.k_proj.weight.shape}"
                    )
                else:
                    logger.info(f"Layer {layer_idx} k_proj has no weight attribute")

        c = None
        if cache is not None and local_idx < len(cache):
            c = cache[local_idx]

        result = self.layers[local_idx](x, mask, c)
        logger.info(f"Layer {layer_idx} output shape: {result.shape}")
        return result

    def _convert_layers_to_quantized(self, weights):
        """Convert Linear layers to QuantizedLinear based on weight structure"""
        if self._converted_to_quantized:
            return

        # Check if weights are quantized by looking for .scales and .biases
        weight_keys = [k for k, _ in weights]
        has_scales = any(".scales" in k for k in weight_keys)
        has_biases = any(".biases" in k for k in weight_keys)

        if not (has_scales and has_biases):
            logger.info("Weights are not quantized, keeping Linear layers")
            return

        logger.info(
            "Detected quantized weights, converting layers to quantized versions"
        )

        from mlx.nn.layers.quantized import QuantizedLinear, QuantizedEmbedding

        # Infer quantization parameters from weight shapes
        # Default to common values if not in config
        group_size = 64
        bits = 8
        if hasattr(self, "quantization_config"):
            group_size = self.quantization_config.get("group_size", 64)
            bits = self.quantization_config.get("bits", 8)

        logger.info(f"Using quantization: bits={bits}, group_size={group_size}")

        # Convert embedding layer for API layer
        if self.is_api_layer and hasattr(self, "embed_tokens"):
            if has_scales and "embed_tokens.scales" in weight_keys:
                # Get the actual dimensions from the quantized weights
                embed_weight = next(
                    (v for k, v in weights if k == "embed_tokens.weight"), None
                )
                if embed_weight is not None:
                    vocab_size = embed_weight.shape[0]
                    # The quantized weight shape is (vocab_size, compressed_dim)
                    # We need to infer the original hidden_size
                    hidden_size = self.config.hidden_size

                    # Use the same group_size as the quantized weights
                    # Infer from scales shape: scales.shape[1] * group_size = hidden_size
                    embed_scales = next(
                        (v for k, v in weights if k == "embed_tokens.scales"), None
                    )
                    if embed_scales is not None:
                        num_groups = embed_scales.shape[1]
                        embed_group_size = hidden_size // num_groups
                    else:
                        embed_group_size = group_size

                    # Replace with QuantizedEmbedding
                    self.embed_tokens = QuantizedEmbedding(
                        vocab_size, hidden_size, group_size=embed_group_size, bits=bits
                    )
                    logger.info(
                        f"Converted embed_tokens to QuantizedEmbedding: {vocab_size}x{hidden_size}, group_size={embed_group_size}"
                    )

        # Convert each layer's Linear modules to QuantizedLinear
        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, "self_attn"):
                # Convert attention layers
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if hasattr(layer.self_attn, proj_name):
                        linear = getattr(layer.self_attn, proj_name)
                        if isinstance(linear, nn.Linear) and not isinstance(
                            linear, QuantizedLinear
                        ):
                            # Create QuantizedLinear with same dimensions
                            in_features = linear.weight.shape[1]
                            out_features = linear.weight.shape[0]
                            ql = QuantizedLinear(
                                in_features,
                                out_features,
                                bias=False,
                                group_size=group_size,
                                bits=bits,
                            )
                            setattr(layer.self_attn, proj_name, ql)
                            logger.debug(
                                f"Converted layer {layer_idx} self_attn.{proj_name} to QuantizedLinear"
                            )

            if hasattr(layer, "mlp"):
                # Convert MLP layers
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    if hasattr(layer.mlp, proj_name):
                        linear = getattr(layer.mlp, proj_name)
                        if isinstance(linear, nn.Linear) and not isinstance(
                            linear, QuantizedLinear
                        ):
                            in_features = linear.weight.shape[1]
                            out_features = linear.weight.shape[0]
                            ql = QuantizedLinear(
                                in_features,
                                out_features,
                                bias=False,
                                group_size=group_size,
                                bits=bits,
                            )
                            setattr(layer.mlp, proj_name, ql)
                            logger.debug(
                                f"Converted layer {layer_idx} mlp.{proj_name} to QuantizedLinear"
                            )

        self._converted_to_quantized = True
        logger.info("Successfully converted all Linear layers to QuantizedLinear")

    def load_weights(self, weights, strict=False):
        """Load weights into the model"""
        logger.info(f"load_weights called with {len(weights)} weights")
        logger.info(f"First few weight keys: {[k for k, _ in weights[:5]]}")
        logger.info(f"abs_to_local mapping: {self.abs_to_local}")

        # Convert layers to quantized if loading quantized weights
        self._convert_layers_to_quantized(weights)

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
                logger.debug(f"Mapping weight {key} (shape {value.shape}) -> {new_key}")
                shard_weights[new_key] = value
            elif self.is_api_layer:
                # API layer needs embed_tokens, norm, and lm_head
                # Weights come as "embed_tokens.weight", "norm.weight", etc.
                if key.startswith("embed_tokens"):
                    shard_weights[key] = value
                    logger.info(f"API layer: loading {key}, shape={value.shape}")
                elif key.startswith("norm"):
                    shard_weights[key] = value
                    logger.info(f"API layer: loading {key}, shape={value.shape}")
                elif key.startswith("lm_head") and not self.config.tie_word_embeddings:
                    shard_weights[key] = value
                    logger.info(f"API layer: loading {key}, shape={value.shape}")

        logger.info(f"Loading {len(shard_weights)} weights into model")

        if shard_weights:
            # Log the first weight being loaded to check dimensions
            first_key = list(shard_weights.keys())[0]
            logger.info(
                f"First weight to load: {first_key} with shape {shard_weights[first_key].shape}"
            )

            # Check what layer this is for
            if "layers." in first_key:
                layer_idx = first_key.split(".")[1]
                logger.info(f"Loading into local layer {layer_idx}")
                logger.info(f"Number of layers in model: {len(self.layers)}")
                if int(layer_idx) < len(self.layers):
                    layer = self.layers[int(layer_idx)]
                    # Log the current layer structure
                    if hasattr(layer, "self_attn"):
                        if hasattr(layer.self_attn, "q_proj"):
                            logger.info(
                                f"Current q_proj type: {type(layer.self_attn.q_proj)}"
                            )
                            if hasattr(layer.self_attn.q_proj, "weight"):
                                logger.info(
                                    f"Current q_proj weight shape: {layer.self_attn.q_proj.weight.shape}"
                                )

        # Load the filtered weights using parent class method
        try:
            super().load_weights(list(shard_weights.items()), strict=strict)
            logger.info("Successfully loaded weights")

            # Verify weights were actually loaded (not just shape but values)
            if shard_weights and "layers." in first_key:
                layer_idx = first_key.split(".")[1]
                if int(layer_idx) < len(self.layers):
                    layer = self.layers[int(layer_idx)]
                    if hasattr(layer, "self_attn") and hasattr(
                        layer.self_attn, "q_proj"
                    ):
                        if hasattr(layer.self_attn.q_proj, "weight"):
                            weight = layer.self_attn.q_proj.weight
                            logger.info(
                                f"After loading - q_proj weight stats: shape={weight.shape}, mean={mx.mean(weight).item():.6f}, std={mx.std(weight).item():.6f}"
                            )
                            # Check if weights are reasonable (not all zeros or random)
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
            logger.error(f"Failed to load weights: {e}")
            logger.error(f"Weight keys: {list(shard_weights.keys())}")
            raise

        # Don't apply quantization for pre-quantized models
        # Pre-quantized models already have QuantizedLinear layers from weight loading

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
