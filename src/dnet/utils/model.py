"""Model metadata and weight loading utilities for dnet."""

import ctypes
import glob
import json
import mmap
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm.utils import get_model_path
from mlx_lm.models import cache

from .serialization import safetensor_dtype_map
from ..ring.model.base import BaseRingModel
from .logger import logger

# REGEX associated with LLM
EMBED_TOKENS_RE = re.compile(r"^model\.embed_tokens\.(.+)$")  # Embedding
LAYERS_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")  # Per-layer
LM_HEAD_RE = re.compile(r"^lm_head\.(.+)$")  # Output head
NORM_RE = re.compile(r"^model\.norm\.(.+)$")  # Normalization


def get_model_layer_name(layer_idx: int, name: str) -> str:
    """Format layer weight name.

    Args:
        layer_idx: Layer index
        name: Weight name

    Returns:
        Formatted name like 'layers.{idx}.{name}'
    """
    return f"layers.{layer_idx}.{name}"


def get_model_embed_tokens_name(name: str) -> str:
    """Format embedding weight name.

    Args:
        name: Weight name

    Returns:
        Formatted name like 'embed_tokens.{name}'
    """
    return f"embed_tokens.{name}"


def get_lm_head_name(name: str) -> str:
    """Format LM head weight name.

    Args:
        name: Weight name

    Returns:
        Formatted name like 'lm_head.{name}'
    """
    return f"lm_head.{name}"


def get_model_norm_name(name: str) -> str:
    """Format normalization weight name.

    Args:
        name: Weight name

    Returns:
        Formatted name like 'norm.{name}'
    """
    return f"norm.{name}"


@dataclass(slots=True, frozen=True)
class TensorInfo:
    """The tensor information stored inside safetensor file."""

    dtype: str
    shape: Tuple[int, ...]
    size_bytes: int
    offset: int
    filename: str


@dataclass(frozen=True)
class ModelMetadata:
    """LLM model metadata"""

    path: Path  # Path to the model on local disk
    weight_info: Dict[int, Dict[str, TensorInfo]]  # Weights of each layer
    embed_tokens: Dict[str, TensorInfo]  # Embedding layer
    lm_head: Dict[str, TensorInfo]  # Output head
    norm: Dict[str, TensorInfo]  # Normalization
    config: Any  # configurations stored in safetensor block

    @cached_property
    def embedding_size(self) -> int:
        """Get embedding size from model metadata.

        Args:
            model_metadata: Model metadata

        Returns:
            Embedding size
        """

        # try to get embedding_size first, fallback to hidden_size
        embedding_size = self.model_config.get("embedding_size")
        if embedding_size is None:
            # try to infer from embed_tokens tensor dimensions
            if self.embed_tokens and "weight" in self.embed_tokens:
                embedding_size = self.embed_tokens["weight"].shape[1]
            else:
                # fallback to hidden_size
                embedding_size = self.model_config.get("hidden_size")

        if embedding_size is None:
            raise ValueError(
                "Could not find embedding_size or hidden_size in model metadata"
            )

        return embedding_size

    @cached_property
    def num_layers(self) -> int:
        """Number of layers (global).

        Prefer explicit value from config (num_hidden_layers) when present so
        that repacked/partial weight sets (e.g., per-shard) still report the
        correct global layer count.
        """
        try:
            n = int(self.config.get("num_hidden_layers"))
            if n > 0:
                return n
        except Exception:
            pass
        return max(self.weight_info.keys()) + 1

    @cached_property
    def model_type(self) -> str:
        """# Model type e.g., llama"""
        return self.config["model_type"]

    @property
    def model_config(self) -> Any:
        return self.config


class MappedFile:
    """Maps a file to memory."""

    def __init__(self, file_path: str, access=mmap.ACCESS_COPY):
        """
        Args:
            file_path: Path to file
        """
        self.file_path = file_path
        # Prefer private copy-on-write mapping so the buffer is writable from
        # Python's perspective (needed for ctypes.from_buffer) without touching disk
        if access == mmap.ACCESS_COPY:
            self.file = open(file_path, "rb")
        else:
            # Fallback path
            self.file = open(file_path, "r+b")

        # Memory map the file
        try:
            self.mmap = mmap.mmap(self.file.fileno(), 0, access=access)
        except Exception:
            # Fallback to copy-on-write if requested access fails
            self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_COPY)

        # Get memory address for madvise
        # from_buffer requires a writable view; ACCESS_COPY satisfies this
        self.base_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mmap))


def load_weight(wt: TensorInfo, mapped_files: Dict[str, MappedFile]) -> mx.array:
    offset, size = wt.offset, wt.size_bytes
    if wt.filename not in mapped_files:
        mapped_files[wt.filename] = MappedFile(wt.filename)
    mapped_file = mapped_files[wt.filename]
    mv = memoryview(mapped_file.mmap)
    layer_data = mv[offset : offset + size]

    # Special handling for BF16
    if wt.dtype == "BF16":
        # BF16 needs special handling - read as uint16 and convert
        uint16_data = np.frombuffer(
            layer_data, dtype=np.uint16
        )  # FIXME: reference before assignment
        float32_data = (uint16_data.astype(np.uint32) << 16).view(np.float32)
        return mx.array(float32_data).reshape(wt.shape).astype(mx.bfloat16)
    else:
        np_data = np.frombuffer(layer_data, dtype=safetensor_dtype_map[wt.dtype])
        return mx.array(np_data).reshape(wt.shape)


def load_embeddings(model_metadata: ModelMetadata, model: BaseRingModel) -> int:
    """Load only embedding weights into the model.

    Returns the number of tensors loaded.
    """
    weights: Dict[str, mx.array] = {}
    mapped_files: Dict[str, MappedFile] = {}
    try:
        for k, wt in model_metadata.embed_tokens.items():
            weights[get_model_embed_tokens_name(k)] = load_weight(wt, mapped_files)
        if weights:
            model.load_weights(list(weights.items()), strict=False)
        return len(weights)
    finally:
        for mapped_file in mapped_files.values():
            mapped_file.mmap.close()
            mapped_file.file.close()


def load_final_norm(model_metadata: ModelMetadata, model: BaseRingModel) -> int:
    """Load only the final normalization weights into the model."""
    weights: Dict[str, mx.array] = {}
    mapped_files: Dict[str, MappedFile] = {}
    try:
        for k, wt in model_metadata.norm.items():
            weights[get_model_norm_name(k)] = load_weight(wt, mapped_files)
        if weights:
            model.load_weights(list(weights.items()), strict=False)
        return len(weights)
    finally:
        for mapped_file in mapped_files.values():
            mapped_file.mmap.close()
            mapped_file.file.close()


def load_lm_head(model_metadata: ModelMetadata, model: BaseRingModel) -> int:
    """Load only the LM head weights into the model.

    Handles both quantized and dense head.
    Returns the number of tensors loaded.
    """
    weights: Dict[str, mx.array] = {}
    mapped_files: Dict[str, MappedFile] = {}
    try:
        hidden_size = getattr(getattr(model, "config", {}), "hidden_size", None)
        vocab_size = getattr(getattr(model, "config", {}), "vocab_size", None)

        lm_keys = set(model_metadata.lm_head.keys())
        has_quant_head = any(k.startswith("scales") for k in lm_keys)

        if has_quant_head:
            for k, wt in model_metadata.lm_head.items():
                weights[get_lm_head_name(k)] = load_weight(wt, mapped_files)
            logger.info("Loaded quantized lm_head params")
        else:
            w_info = model_metadata.lm_head.get("weight")
            if (
                w_info is not None
                and hidden_size is not None
                and vocab_size is not None
            ):
                w_arr = load_weight(w_info, mapped_files)
                shp = tuple(w_arr.shape)
                if shp == (hidden_size, vocab_size):
                    weights[get_lm_head_name("weight")] = w_arr
                    logger.info("Loaded dense lm_head.weight")
                elif shp == (vocab_size, hidden_size):
                    weights[get_lm_head_name("weight")] = w_arr.T
                    logger.info("Loaded transposed dense lm_head.weight")
                else:
                    logger.warning(
                        "Skipping lm_head.weight with incompatible shape %s; using tied projection if configured.",
                        shp,
                    )
            else:
                if w_info is None:
                    logger.info("No lm_head.weight found; tied projection may be used")

        if weights:
            model.load_weights(list(weights.items()), strict=False)
        return len(weights)
    finally:
        for mapped_file in mapped_files.values():
            mapped_file.mmap.close()
            mapped_file.file.close()


def load_api_layer_weights(model_metadata: ModelMetadata, model: BaseRingModel):
    """Deprecated wrapper: Loads embeddings, norm, and LM head (legacy behavior).

    Prefer calling load_embeddings/load_final_norm/load_lm_head selectively
    based on the shard's role and tie_word_embeddings config.
    """
    # Legacy combined load
    cnt = 0
    cnt += load_embeddings(model_metadata, model)
    cnt += load_final_norm(model_metadata, model)
    # For head, respect tied setting if model exposes it
    tied = bool(getattr(getattr(model, "config", object()), "tie_word_embeddings", False))
    if not tied:
        cnt += load_lm_head(model_metadata, model)
    else:
        try:
            setattr(model, "force_tied_head", True)
        except Exception:
            pass
    try:
        model.eval()
    except Exception:
        pass
    logger.info("Loaded %d API-layer tensors via legacy loader", cnt)


def get_safetensor_details(path) -> Dict[str, TensorInfo]:
    with open(path, "rb") as f:
        # Memory-map file
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Read header length (first 8 bytes, little-endian u64)
        header_len = struct.unpack("<Q", mm[:8])[0]

        # Read header JSON
        header_bytes = mm[8 : 8 + header_len]
        header = json.loads(header_bytes.decode("utf-8"))

        # Compute base offset for tensor data
        data_base = 8 + header_len

        details = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue

            start, end = info["data_offsets"]
            details[name] = TensorInfo(
                dtype=info["dtype"],
                shape=info["shape"],
                size_bytes=end - start,
                offset=data_base + start,
                filename=path,
            )

        return details


def get_model_metadata(model_path) -> ModelMetadata:
    path = get_model_path(model_path)

    # Handle case where get_model_path returns a tuple (mlx-lm version compatibility)
    if isinstance(path, tuple):
        path = path[0] if path else None
    if path is None:
        raise ValueError(f"Could not resolve model path for {model_path}")

    path = Path(path) if not isinstance(path, Path) else path

    # read model configurations
    with open(path / "config.json", "r") as f:
        config = json.load(f)

    weight_files = glob.glob(str(path / "*.safetensors"))
    weight_info: Dict[int, Dict[str, Any]] = defaultdict(dict)
    embed_tokens, lm_head, norm = {}, {}, {}
    for weight in weight_files:
        details = get_safetensor_details(weight)
        for key, val in details.items():
            if m := EMBED_TOKENS_RE.match(key):
                embed_tokens[m.group(1)] = val
            elif m := LM_HEAD_RE.match(key):
                lm_head[m.group(1)] = val
            elif m := NORM_RE.match(key):
                norm[m.group(1)] = val
            elif m := LAYERS_RE.match(key):
                layer_idx, suffix = m.groups()
                weight_info[int(layer_idx)][suffix] = val
            else:
                raise RuntimeError(f"Unexpected key {key}")
    # Allow partial/repacked weight sets for shard-local models:
    # - Do not require contiguous layer coverage [0..max].
    # - If the global layer count is available in config (num_hidden_layers),
    #   validate that any present layer indices fall within [0, num_hidden_layers).
    try:
        cfg_layers = int(config.get("num_hidden_layers", -1))
    except Exception:
        cfg_layers = -1
    if cfg_layers > 0:
        bad = [i for i in weight_info if i < 0 or i >= cfg_layers]
        if bad:
            raise RuntimeError(
                f"Layer indices out of range for model (num_hidden_layers={cfg_layers}): {sorted(set(bad))}"
            )

    return ModelMetadata(path, weight_info, embed_tokens, lm_head, norm, config)


def make_cache(
    model: BaseRingModel,
    *,
    kv_mode: str | None = None,
    kv_bits: int | None = None,
    kv_group: int | None = None,
):
    """Create model KV cache with optional quantization (config-only).

    This function does not read environment variables. Callers must pass
    kv_mode/bits/group explicitly, or rely on the defaults below.
    """
    caches = cache.make_prompt_cache(model)

    # Resolve mode strictly from parameters (no env)
    mode: str = (kv_mode or "fp16").strip().lower()

    if mode in {"int8", "int4", "quant", "q"}:
        bits_env = int(kv_bits if kv_bits is not None else 8)
        # Map mode shortcuts
        if mode == "int4":
            bits = 4
        elif mode == "int8":
            bits = 8
        else:
            bits = max(1, min(8, bits_env))
        group = int(kv_group if kv_group is not None else 64)

        converted = []
        converted_any = False
        for c in caches:
            if hasattr(c, "to_quantized"):
                try:
                    qc = c.to_quantized(group_size=group, bits=bits)  # type: ignore[attr-defined]
                    converted.append(qc)
                    converted_any = True
                except Exception as e:
                    logger.warning(
                        "KV quantization failed for one cache entry: %s; using fp16 entry",
                        e,
                    )
                    converted.append(c)
            else:
                converted.append(c)

        if converted_any:
            logger.info(
                "Enabled quantized KV cache: bits=%s, group_size=%s", bits, group
            )
            return converted
        else:
            logger.info(
                "KV quantization requested but not supported by cache type; using fp16 KV cache"
            )
            return caches

    # Default fp16/unquantized cache
    return caches
