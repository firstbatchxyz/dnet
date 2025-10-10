"""Tensor serialization utilities for dnet."""

import mlx.core as mx
import numpy as np


# Dtype mapping for numpy types (used for serialization)
dtype_map = {
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
    # Treat BF16 payload as float16 for NumPy buffer interpretation
    "mlx.core.bfloat16": np.float16,
}

safetensor_dtype_map = {
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
    "BF16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
}

mlx_dtype_map = {
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
