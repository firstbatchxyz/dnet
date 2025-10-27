from typing import Any, Dict, List, Optional, Tuple

import mlx.nn as nn
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.llama import ModelArgs, TransformerBlock 

from .base import BaseRingModel

import logging
logger = logging.getLogger(__name__)


class Llama3RingModel(BaseRingModel):
  model_type = "llama"

  def __init__(
    self, 
    model_config: Any,
    assigned_layers: Optional[List[int]] = [],
    is_api_layer: bool = False,
    shard_config: Optional[Any] = None,
  ):
    super().__init__()

    if is_api_layer and assigned_layers:
      raise RuntimeError(f"API Service doesn't handle layers")

    self.config = ModelArgs.from_dict(model_config)
    self.config.quantization = model_config["quantization"] # lmao
    self.is_api_layer = is_api_layer

    self._converted_to_quantized = False
    self.runtime_cache: Optional[List[Any]] = None

    self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
    self.norm = nn.RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)

    if not self.config.tie_word_embeddings:
      self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    self.layers: List[nn.Module] = []
    self.abs_to_local: Dict[int, int] = {}

    for i, l in enumerate(sorted(assigned_layers or [])):
      self.layers.append(TransformerBlock(self.config))
      self.abs_to_local[l] = i

    logger.debug(f"Created {len(self.layers)} Transformer layers")
    #logger.debug(f"abs_to_local mapping: {self.abs_to_local}")

  @property
  def decoding_layers(self):
    return self.layers

  @property
  def head_dim(self) -> Tuple[int, int]:
    return self.config.head_dim

  @property
  def n_kv_heads(self) -> int:
    return self.config.num_key_value_heads

  @property
  def num_layers(self) -> int:
    return len(self.layers)

  def set_runtime_cache(self, cache: Optional[List[Any]]) -> None:
    self._runtime_cache = cache

  def class_predicate(p, m):
    return hasattr(m, "to_quantized")

  def embed(self, x: mx.array):
    return self.embed_tokens(x) 

  def normalize(self, x: mx.array):
    return self.norm(x) 

  # FIXME: Weird MLX bug, lm_head weights are transposed internally for no reason 
  def lm_project(self, x: mx.array):
    if self.config.tie_word_embeddings:
      return self.embed_tokens.as_linear(x)
    try:
      return self.lm_head(x)
    except Exception as e:
      return mx.matmul(x, self.lm_head.weight)

  def quantize_layers(self):
    self.quantization = None 
    logger.debug(f"{self.config}")
    if hasattr(self.config, "quantization"):
      self.quantization = getattr(self.config, "quantization")
    elif hasattr(self.config, "quantization_config"):
      self.quantization = getattr(self.config, "quantization_config")

    logger.debug(f"QUANTIZING {self.quantization}")
    if self.quantization is not None:
      bits = int(self.quantization.get("bits", 4))
      group = int(self.quantization.get("group_size", 64))
      try:
        from mlx.nn.layers.quantized import QuantizedEmbedding
        self.embed_tokens = QuantizedEmbedding(self.config.vocab_size, 
                                               self.config.hidden_size,
                                               group_size=group, bits=bits)

        logger.debug(f"API Service initialized to QuantizedEmbedding:" 
                     f"{self.config.vocab_size}, hidden={self.config.hidden_size}"
                     f"group_size={group}, bits={bits}")
      except Exception as e:
        logger.warning(f"Unable to initialize QuantizedEmbedding: {e}")

      try:
        nn.quantize(self, bits=bits, group_size=group, class_predicate=Llama3RingModel.class_predicate)
        logger.debug(f"Quantized the model: bits={bits}, group_size={group}")
        self._converted_to_quantized = True
      except:
        self._converted_to_quantized = False 

  def forward(
    self, 
    x: mx.array, 
    cache: Optional[List[Any]] = None
  ):
    mask = create_attention_mask(x, cache)
    if cache is None:
      cache = [None] * len(self.layers)

    for i, l in enumerate(self.layers):
      x = l(x, mask, cache[i] if i < len(cache) else None)

    return x
    
  # TODO: Original implementation is slidin window. Bench to see if it's faster or just do sparse
  def apply_single_layer(
    self,
    layer_idx: int, 
    x: mx.array,
    cache: Optional[List[Any]] = None
  ):
    if layer_idx not in self.abs_to_local:
      raise RuntimeError(f"Attempted execution of foreign layer {layer_idx}")

    mask = None
    sqlen = int(x.shape[1])
    if sqlen > 1:
      cached = getattr(self, "_cached_mask_len", None)  
      cached_mask = getattr(self, "_cached_mask", None)
      if cached is None or cached != sqlen or not cached_mask:
        mask = create_attention_mask(x, cache)
        self._cached_mask = mask
        self._cached_mask_len = sqlen
      else:
        mask = cached_mask
        
    local_idx = self.abs_to_local[layer_idx]
    logger.debug(f"apply_single_layer: layer:{layer_idx}, local_idx:{local_idx}, input_shape:{x.shape}")

    layer = self.layers[local_idx]
    ret = self.layers[local_idx](x, mask, cache[local_idx] if local_idx < len(cache) else None)
    return ret

  def load_weights(self, weights, strict=False):
    weight_keys = [k for k, _ in weights]
    has_scales = any(".scales" in k for k in weight_keys)
    has_biases = any(".biases" in k for k in weight_keys)

    if has_scales and has_biases:
      if not self._converted_to_quantized:
        self.quantize_layers()

    shard_weights = {}
    for k, v in weights:
      if k.startswith("model.layers.") or k.startswith("layers."):
        p = k.split(".")
        idx_pos = 2 if p[0] == "model" else 1
        try:
          idx = int(p[idx_pos])
        except Exception as e:
          logger.warning(f"Unable to read weight positions: {e}")
          continue
        if idx not in self.abs_to_local:
          continue
        local_idx = self.abs_to_local[idx]
        p[idx_pos] = str(local_idx)
        if p[0] == "model":
          p = p[1:]
        new_key = ".".join(p)
        logger.debug(f"Mapping weight {k} -> {new_key}")
        shard_weights[new_key] = v
        
      elif k.startswith("lm_head"):
        shard_weights[k] = v
      elif (k.startswith("embed_tokens") or k.startswith("norm")):
        shard_weights[k] = v
        
    if shard_weights:
      try:
        super().load_weights(list(shard_weights.items()), strict=strict)
        logger.debug(f"Loaded {len(shard_weights.keys())} weights into model")    
      except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        logger.error(f"Weight keys: {list(shard_weights.keys())}")
        raise

  def unload_layers(self, layers: List[int]):
    for l in layers:
      local = self.abs_to_local[l]
      for name, mod in self.layers[local].named_modules():
        if name in ['self_attn', 'mlp']:
          for pname in mod.parameters():
            setattr(mod, pname, None)
            logger.debug(f"Unloaded {pname}")
        elif name in ['input_layernorm', 'post_attention_layernorm']:
          mod.weight = None
          logger.debug(f"Unloaded {name}")
