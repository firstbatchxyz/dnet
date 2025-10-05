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
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm.utils import get_model_path

from .serialization import safetensor_dtype_map


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
    """LLM model metadata."""

    path: Path  # Path to the model on local disk
    weight_info: Dict[int, Dict[str, TensorInfo]]  # Weights of each layer
    embed_tokens: Dict[str, TensorInfo]  # Embedding layer
    lm_head: Dict[str, TensorInfo]  # Output head
    norm: Dict[str, TensorInfo]  # Normalization
    config: Any  # configurations stored in safetensor block

    @cached_property
    def num_layers(self) -> int:
        """Number of layers in the model."""
        return max(self.weight_info.keys()) + 1

    @cached_property
    def model_type(self) -> str:
        """Model type (e.g., 'qwen3', 'deepseek_v2')."""
        return self.config["model_type"]

    @property
    def model_config(self) -> Any:
        """Model configuration dict."""
        return self.config


class MappedFile:
    """Maps a file to memory for efficient weight loading."""

    def __init__(self, file_path: str, access=mmap.ACCESS_WRITE):
        """Initialize memory-mapped file.

        Args:
            file_path: Path to file
            access: mmap access mode
        """
        self.file_path = file_path
        self.file = open(file_path, "r+b")

        # Memory map the file
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=access)

        # Get memory address for madvise
        self.base_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mmap))


def load_weight(wt: TensorInfo, mapped_files: Dict[str, MappedFile]) -> mx.array:
    """Load a single weight tensor from memory-mapped file.

    Args:
        wt: Tensor information
        mapped_files: Dictionary of already-mapped files

    Returns:
        Loaded MLX array
    """
    offset, size = wt.offset, wt.size_bytes

    if wt.filename not in mapped_files:
        mapped_files[wt.filename] = MappedFile(wt.filename)

    mapped_file = mapped_files[wt.filename]
    layer_data = mapped_file.mmap[offset : offset + size]

    # Special handling for BF16
    if wt.dtype == "BF16":
        # BF16 needs special handling - read as uint16 and convert
        # BF16 is stored as 16-bit values
        uint16_data = np.frombuffer(layer_data, dtype=np.uint16)
        # Convert uint16 to float32 by interpreting as bfloat16
        # BF16: sign(1) + exponent(8) + mantissa(7)
        # Move the bits to the upper 16 bits of float32
        float32_data = (uint16_data.astype(np.uint32) << 16).view(np.float32)
        return mx.array(float32_data).reshape(wt.shape).astype(mx.bfloat16)
    else:
        np_data = np.frombuffer(layer_data, dtype=safetensor_dtype_map[wt.dtype])
        return mx.array(np_data).reshape(wt.shape)


def load_api_layer_weights(model_metadata: ModelMetadata, model: Any) -> None:
    """Load API layer weights (embeddings, norm, lm_head) into model.

    Args:
        model_metadata: Model metadata containing weight information
        model: Model instance to load weights into
    """
    weights: Dict[str, mx.array] = {}
    mapped_files: Dict[str, MappedFile] = {}

    try:
        for k, wt in model_metadata.embed_tokens.items():
            weights[get_model_embed_tokens_name(k)] = load_weight(wt, mapped_files)

        for k, wt in model_metadata.lm_head.items():
            weights[get_lm_head_name(k)] = load_weight(wt, mapped_files)

        for k, wt in model_metadata.norm.items():
            weights[get_model_norm_name(k)] = load_weight(wt, mapped_files)

        model.load_weights(list(weights.items()))
        model.eval()
    finally:
        for mapped_file in mapped_files.values():
            mapped_file.mmap.close()
            mapped_file.file.close()


def get_safetensor_details(path: str) -> Dict[str, TensorInfo]:
    """Extract tensor details from a safetensor file.

    Args:
        path: Path to safetensor file

    Returns:
        Dictionary mapping tensor names to TensorInfo
    """
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

        details: Dict[str, TensorInfo] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue

            start, end = info["data_offsets"]
            details[name] = TensorInfo(
                dtype=info["dtype"],
                shape=tuple(info["shape"]),
                size_bytes=end - start,
                offset=data_base + start,
                filename=path,
            )

        return details


def get_model_metadata(model_path: str) -> ModelMetadata:
    """Load model metadata from a HuggingFace model path.

    Args:
        model_path: Path or HuggingFace repo ID

    Returns:
        ModelMetadata instance

    Raises:
        ValueError: If model path cannot be resolved
        RuntimeError: If weights are inconsistent
    """
    path = get_model_path(model_path)

    # Handle case where get_model_path returns a tuple (mlx-lm version compatibility)
    if isinstance(path, tuple):
        path = path[0] if path else None
    if path is None:
        raise ValueError(f"Could not resolve model path for {model_path}")

    # Ensure it's a Path object
    path = Path(path) if not isinstance(path, Path) else path

    # read model configurations
    with open(path / "config.json", "r") as f:
        config = json.load(f)

    weight_files = glob.glob(str(path / "*.safetensors"))
    weight_info: Dict[int, Dict[str, TensorInfo]] = defaultdict(dict)
    embed_tokens: Dict[str, TensorInfo] = {}
    lm_head: Dict[str, TensorInfo] = {}
    norm: Dict[str, TensorInfo] = {}

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

    num_layers = max(weight_info.keys()) + 1
    if not (set(weight_info.keys()) == set(range(num_layers))):
        raise RuntimeError("Inconsistent weights")

    return ModelMetadata(path, weight_info, embed_tokens, lm_head, norm, config)


def get_layer_assignment_rr(num_nodes: int, metadata: ModelMetadata) -> Dict[int, List[int]]:
    """Get round-robin layer assignment across nodes.

    Args:
        num_nodes: Number of nodes
        metadata: Model metadata

    Returns:
        Dictionary mapping node index to list of layer indices
    """
    from itertools import cycle

    num_layers = metadata.num_layers
    partitions: Dict[int, List[int]] = defaultdict(list)
    for idx, x in zip(cycle(range(num_nodes)), range(num_layers)):
        partitions[idx].append(x)
    return dict(partitions)
