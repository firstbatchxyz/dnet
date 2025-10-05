"""Tensor serialization utilities for dnet."""

import time
from typing import Dict

import mlx.core as mx
import numpy as np


# Dtype mapping for numpy types (used for serialization)
dtype_map: Dict[str, type[np.generic]] = {
    "mlx.core.float32": np.float32,
    "mlx.core.int32": np.int32,
    "mlx.core.int64": np.int64,
    "mlx.core.float16": np.float16,
    "mlx.core.int8": np.int8,
    "mlx.core.int16": np.int16,
    "mlx.core.uint8": np.uint8,
    "mlx.core.uint16": np.uint16,
    "mlx.core.uint32": np.uint32,
    "mlx.core.uint64": np.uint64,
    "mlx.core.bfloat16": np.float32,  # BF16 is not directly supported by numpy, use float32
}

# Safetensor dtype mapping
safetensor_dtype_map: Dict[str, type[np.generic]] = {
    "BOOL": np.bool_,
    "U8": np.uint8,
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "F16": np.float16,
    "BF16": np.float16,  # Map BF16 to F16 as numpy doesn't have native bfloat16
    "F32": np.float32,
    "F64": np.float64,
}

# MLX dtype mapping
mlx_dtype_map: Dict[str, mx.Dtype] = {
    "mlx.core.float32": mx.float32,
    "mlx.core.int32": mx.int32,
    "mlx.core.int64": mx.int64,
    "mlx.core.float16": mx.float16,
    "mlx.core.int8": mx.int8,
    "mlx.core.int16": mx.int16,
    "mlx.core.uint8": mx.uint8,
    "mlx.core.uint16": mx.uint16,
    "mlx.core.uint32": mx.uint32,
    "mlx.core.uint64": mx.uint64,
    "mlx.core.bfloat16": mx.bfloat16,
}


def tensor_to_bytes(tensor: mx.array) -> bytes:
    """Convert MLX tensor to bytes.

    Args:
        tensor: MLX array to serialize

    Returns:
        Bytes representation of the tensor
    """
    return bytes(memoryview(tensor))


def bytes_to_tensor(byte_data: bytes, dtype_str: str) -> mx.array:
    """Convert bytes back to MLX tensor.

    Args:
        byte_data: Serialized tensor bytes
        dtype_str: String representation of dtype (e.g., "mlx.core.float32")

    Returns:
        Deserialized MLX array

    Raises:
        ValueError: If dtype is unsupported
    """
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    np_dtype = dtype_map.get(dtype_str, np.float32)
    np_array = np.frombuffer(byte_data, dtype=np_dtype)

    mx_dtype_str = dtype_str.replace("mlx.core.", "")
    mx_dtype = getattr(mx, mx_dtype_str, mx.float32)

    return mx.array(np_array, dtype=mx_dtype)


def utc_epoch_now() -> int:
    """Get current UTC epoch time in milliseconds.

    High-resolution UTC epoch in milliseconds as int.
    Previous implementation used whole seconds, which quantized
    transport timing to ~1000 ms buckets and obscured latency.

    Returns:
        Current time in milliseconds since epoch
    """
    return int(time.time() * 1000)
