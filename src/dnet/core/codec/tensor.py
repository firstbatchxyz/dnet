import numpy as np
import mlx.core as mx
from dnet.utils.serialization import dtype_map, tensor_to_bytes


def to_bytes(
    tensor: mx.array | np.ndarray,
    *,
    wire_dtype_str: str,
    wire_mx_dtype: mx.Dtype,
    compress: bool = False,
    compress_min_bytes: int = 65536,
) -> bytes:
    """Serialize an MLX/Numpy tensor to bytes with the given wire dtype.

    Args:
        tensor: MLX or NumPy array
        wire_dtype_str: Canonical dtype string (e.g., "float16", "bfloat16")
        wire_mx_dtype: MLX dtype to cast to when `tensor` is MLX
        compress: Whether to compress payload (currently not applied)
        compress_min_bytes: Minimum size for compression to kick in

    Returns:
        bytes: Serialized tensor data
    """
    # NB: Compression is intentionally disabled for decode path; keep parity.
    _ = compress
    _ = compress_min_bytes

    # Cast to desired wire dtype without extra copies when possible
    try:
        wire_np_dtype = dtype_map[wire_dtype_str]
    except Exception:
        wire_np_dtype = np.float16

    if isinstance(tensor, np.ndarray):
        if tensor.dtype != wire_np_dtype:
            tensor = tensor.astype(wire_np_dtype, copy=False)
    else:
        if str(tensor.dtype) != wire_dtype_str:
            tensor = tensor.astype(wire_mx_dtype)

    if isinstance(tensor, np.ndarray):
        data = tensor.tobytes(order="C")
    else:
        data = tensor_to_bytes(tensor)

    return data
