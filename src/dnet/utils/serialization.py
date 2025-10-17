"""Tensor serialization utilities for dnet."""

import mlx.core as mx
import numpy as np

try:
    NP_BF16 = np.dtype("bfloat16")
except Exception:
    NP_BF16 = None

# Canonical -> aliases
ALIASES = {
    "bfloat16": ("mlx.core.bfloat16", "bf16", "BF16"),
    "float16": ("mlx.core.float16", "f16", "F16"),
    "float32": ("mlx.core.float32", "f32", "F32"),
    "float64": ("mlx.core.float64", "f64", "F64"),
    "int8": ("mlx.core.int8", "i8", "I8"),
    "int16": ("mlx.core.int16", "i16", "I16"),
    "int32": ("mlx.core.int32", "i32", "I32"),
    "int64": ("mlx.core.int64", "i64", "I64"),
    "uint8": ("mlx.core.uint8", "u8", "U8"),
    "uint16": ("mlx.core.uint16", "u16", "U16"),
    "uint32": ("mlx.core.uint32", "u32", "U32"),
    "uint64": ("mlx.core.uint64", "u64", "U64"),
    "bool": ("mlx.core.bool", "BOOL"),
}

NP_MAP_BASE = {
    "bfloat16": (NP_BF16 or np.uint16),  # fallback keeps size=2
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "bool": np.bool_,
}

MX_MAP = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
    "float64": mx.float64,
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int64,
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "uint64": mx.uint64,
}


def _canon(s: str) -> str:
    s = (s or "").strip()
    if s in NP_MAP_BASE:
        return s
    for kn, al in ALIASES.items():
        if s == kn or s in al:
            return kn
    return s


dtype_map = {}
for k, npdt in NP_MAP_BASE.items():
    dtype_map[k] = npdt
    for a in ALIASES[k]:
        dtype_map[a] = npdt

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
    "BF16": (NP_BF16 or np.uint16),  # keep width if no bfloat16
    "F32": np.float32,
    "F64": np.float64,
}

mlx_dtype_map = {}
for k, mxdt in MX_MAP.items():
    mlx_dtype_map[k] = mxdt
    for a in ALIASES[k]:
        if a.islower() or a.startswith("mlx.core."):
            mlx_dtype_map[a] = mxdt


def tensor_to_bytes(tensor: mx.array) -> bytes:
    """
    Serialize MLX tensor to raw bytes.
    """
    return bytes(memoryview(tensor))


def bytes_to_tensor(byte_data: bytes, dtype_str: str) -> mx.array:
    """
    Deserialize raw bytes to MLX tensor of given dtype.
    """
    key = _canon(dtype_str)
    if key not in NP_MAP_BASE:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    # Accurate BF16 decode even when NumPy lacks bfloat16
    if key == "bfloat16" and NP_BF16 is None:
        u16 = np.frombuffer(byte_data, dtype=np.uint16)
        # reinterpret BF16 -> F32 by shifting to high 16 bits
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return mx.array(f32, dtype=mx.bfloat16)
    np_arr = np.frombuffer(byte_data, dtype=NP_MAP_BASE[key])
    return mx.array(np_arr, dtype=MX_MAP.get(key, mx.float32))
