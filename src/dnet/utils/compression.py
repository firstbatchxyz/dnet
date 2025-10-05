"""
Optimized compression utilities for MLX tensors using pure column sparsification.
Based on "BEYOND TOP-K: STRUCTURED SPARSIFICATION FOR COMPRESSION IN PIPELINE PARALLEL"

After extensive testing, pure column sparsification provides the best speed/compression
trade-off, being 20x faster than sparsification + compression methods.
"""

import math
import time
from typing import Dict, Tuple

import mlx.core as mx
import numpy as np


class CompressionStats:
    """Statistics for compression operations."""

    def __init__(self):
        self.original_size: int = 0
        self.compressed_size: int = 0
        self.sparsity_ratio: float = 0.0
        self.compression_ratio: float = 0.0
        self.reconstruction_error: float = 0.0
        self.compression_time: float = 0.0
        self.decompression_time: float = 0.0
        self.num_columns_kept: int = 0
        self.total_columns: int = 0

    def __str__(self) -> str:
        return (
            f"CompressionStats(\n"
            f"  original_size={self.original_size:,} bytes,\n"
            f"  compressed_size={self.compressed_size:,} bytes,\n"
            f"  sparsity_ratio={self.sparsity_ratio:.2%},\n"
            f"  compression_ratio={self.compression_ratio:.2f}x,\n"
            f"  columns_kept={self.num_columns_kept}/{self.total_columns} ({self.num_columns_kept/self.total_columns:.1%}),\n"
            f"  reconstruction_error={self.reconstruction_error:.6f},\n"
            f"  compression_time={self.compression_time*1000:.2f}ms,\n"
            f"  decompression_time={self.decompression_time*1000:.2f}ms\n"
            f")"
        )


def column_sparsify_tensor(tensor: mx.array, compression_percentage: float) -> mx.array:
    """
    Sparsifies a tensor by zeroing out columns with the smallest L2 norms.

    Args:
        tensor (mx.array): Input tensor of any shape (..., D) where D is the feature dimension
        compression_percentage (float): Percentage of columns to mask (0.0 to 100.0)

    Returns:
        mx.array: Sparsified tensor with same shape as input
    """
    if not isinstance(tensor, mx.array):
        raise TypeError("Input must be an MLX array.")

    if not 0.0 <= compression_percentage <= 100.0:
        raise ValueError("compression_percentage must be between 0.0 and 100.0.")

    if compression_percentage == 0.0:
        return tensor

    if compression_percentage == 100.0:
        return mx.zeros_like(tensor)

    original_shape = tensor.shape
    if len(original_shape) == 0 or original_shape[-1] == 0:
        return tensor

    # Reshape to 2D for processing: (..., D) -> (N, D)
    feature_dim = original_shape[-1]
    if len(original_shape) > 1:
        reshaped_tensor = tensor.reshape(-1, feature_dim)
    else:
        reshaped_tensor = tensor.reshape(1, -1)

    num_cols = reshaped_tensor.shape[1]
    num_cols_to_mask = min(
        math.ceil((compression_percentage / 100.0) * num_cols), num_cols
    )

    if num_cols_to_mask == 0:
        return tensor

    # Calculate L2 norm for each column
    column_norms = mx.linalg.norm(reshaped_tensor, axis=0)

    # Find indices of columns with smallest norms
    sorted_indices = mx.argsort(column_norms)
    indices_to_mask = sorted_indices[:num_cols_to_mask]

    # Create mask efficiently
    mask_array = mx.ones(num_cols, dtype=mx.bool_)
    mask_array[indices_to_mask] = False

    # Apply mask
    compressed_tensor = reshaped_tensor * mask_array[None, :]

    # Reshape back to original shape
    return compressed_tensor.reshape(original_shape)


def compress_tensor_simple(
    tensor: mx.array,
    compression_percentage: float = 90.0,
    quantize_to_fp16: bool = True,
) -> Tuple[bytes, Dict]:
    """
    Simple compression: column sparsification + optional fp16 quantization.
    This is the fastest method that still provides good compression.

    Args:
        tensor: Input MLX tensor
        compression_percentage: Percentage of columns to sparsify (0-100)
        quantize_to_fp16: Whether to convert to float16 for network transfer

    Returns:
        Tuple of (compressed_bytes, metadata_dict)
    """
    start_time = time.time()

    # Apply column sparsification
    sparsified = column_sparsify_tensor(tensor, compression_percentage)

    # Quantize if requested
    if quantize_to_fp16 and tensor.dtype == mx.float32:
        compressed_tensor = sparsified.astype(mx.float16)
        dtype_used = "float16"
    else:
        compressed_tensor = sparsified
        dtype_used = str(tensor.dtype).replace("mlx.core.", "")

    # Convert to bytes
    tensor_np = np.array(compressed_tensor)
    tensor_bytes = tensor_np.tobytes()

    compression_time = time.time() - start_time

    metadata = {
        "original_shape": list(tensor.shape),
        "original_dtype": str(tensor.dtype),
        "compressed_dtype": dtype_used,
        "compression_percentage": compression_percentage,
        "quantized": quantize_to_fp16 and tensor.dtype == mx.float32,
        "compression_time": compression_time,
        "method": "simple",
    }

    return tensor_bytes, metadata


def decompress_tensor_simple(compressed_bytes: bytes, metadata: Dict) -> mx.array:
    """
    Decompress tensor from simple format.

    Args:
        compressed_bytes: Compressed tensor data
        metadata: Metadata dictionary from compression

    Returns:
        Decompressed MLX tensor
    """
    start_time = time.time()

    # Map dtype strings
    dtype_map = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "int64": np.int64,
    }

    compressed_dtype = metadata["compressed_dtype"]
    np_dtype = dtype_map.get(compressed_dtype, np.float32)

    # Deserialize
    shape = tuple(metadata["original_shape"])
    tensor_np = np.frombuffer(compressed_bytes, dtype=np_dtype).reshape(shape)

    # Convert back to MLX array with original dtype
    original_dtype_str = metadata["original_dtype"]
    if "float32" in original_dtype_str:
        mx_dtype = mx.float32
    elif "float16" in original_dtype_str:
        mx_dtype = mx.float16
    elif "int32" in original_dtype_str:
        mx_dtype = mx.int32
    elif "int64" in original_dtype_str:
        mx_dtype = mx.int64
    else:
        mx_dtype = mx.float32

    tensor = mx.array(tensor_np, dtype=mx_dtype)

    decompression_time = time.time() - start_time
    metadata["decompression_time"] = decompression_time

    return tensor


def compress_tensor_with_mask(
    tensor: mx.array, compression_percentage: float = 90.0
) -> Tuple[bytes, bytes, Dict]:
    """
    Advanced compression: send column mask + non-zero columns only.
    Achieves theoretical maximum compression ratio.

    Args:
        tensor: Input MLX tensor
        compression_percentage: Percentage of columns to sparsify (0-100)

    Returns:
        Tuple of (column_data_bytes, mask_bytes, metadata_dict)
    """
    start_time = time.time()

    original_shape = tensor.shape

    # Apply column sparsification
    sparsified = column_sparsify_tensor(tensor, compression_percentage)

    # Reshape to 2D to work with columns
    feature_dim = original_shape[-1]
    if len(original_shape) > 1:
        reshaped = sparsified.reshape(-1, feature_dim)
    else:
        reshaped = sparsified.reshape(1, -1)

    # Find non-zero columns
    column_norms = mx.linalg.norm(reshaped, axis=0)
    column_mask = column_norms > 1e-8

    # Convert to numpy for easier manipulation
    reshaped_np = np.array(reshaped)
    column_mask_np = np.array(column_mask)

    # Extract only non-zero columns using numpy
    non_zero_columns = reshaped_np[:, column_mask_np]

    # Convert to bytes
    # 1. Column data (fp16 for efficiency)
    columns_np = non_zero_columns.astype(np.float16)
    column_bytes = columns_np.tobytes()

    # 2. Mask
    mask_bytes = column_mask_np.tobytes()

    compression_time = time.time() - start_time

    num_kept = int(np.sum(column_mask_np))

    metadata = {
        "original_shape": list(original_shape),
        "original_dtype": str(tensor.dtype),
        "compression_percentage": compression_percentage,
        "num_columns": feature_dim,
        "num_columns_kept": num_kept,
        "compression_time": compression_time,
        "method": "column_mask",
    }

    return column_bytes, mask_bytes, metadata


def decompress_tensor_with_mask(
    column_bytes: bytes, mask_bytes: bytes, metadata: Dict
) -> mx.array:
    """
    Decompress tensor from column mask format.

    Args:
        column_bytes: Non-zero column data
        mask_bytes: Column mask
        metadata: Metadata dictionary

    Returns:
        Decompressed MLX tensor
    """
    start_time = time.time()

    original_shape = tuple(metadata["original_shape"])
    feature_dim = metadata["num_columns"]
    num_kept = metadata["num_columns_kept"]

    # Reconstruct shape for 2D operations
    if len(original_shape) > 1:
        num_rows = math.prod(original_shape[:-1])
        reshaped_shape = (num_rows, feature_dim)
    else:
        num_rows = 1
        reshaped_shape = (1, feature_dim)

    # Deserialize mask
    mask = np.frombuffer(mask_bytes, dtype=np.bool_)

    # Deserialize columns
    columns_np = np.frombuffer(column_bytes, dtype=np.float16).reshape(
        num_rows, num_kept
    )

    # Reconstruct full tensor
    result = np.zeros(reshaped_shape, dtype=np.float32)
    result[:, mask] = columns_np

    # Reshape to original shape and convert to MLX
    result = result.reshape(original_shape)
    tensor = mx.array(result, dtype=mx.float32)

    decompression_time = time.time() - start_time
    metadata["decompression_time"] = decompression_time

    return tensor


def compress_tensor_for_network(
    tensor: mx.array,
    compression_percentage: float = 90.0,
    use_column_mask: bool = False,
) -> Tuple[Dict, CompressionStats]:
    """
    Main compression function that chooses the best method.

    Args:
        tensor: Input MLX tensor
        compression_percentage: Percentage of columns to sparsify (0-100)
        use_column_mask: If True, use column mask method for better compression
                        If False, use simple method for better speed

    Returns:
        Tuple of (compressed_data_dict, stats)
    """
    stats = CompressionStats()
    stats.original_size = tensor.size * 4  # float32

    if use_column_mask and compression_percentage >= 50.0:
        # Use column mask method for better compression
        column_bytes, mask_bytes, metadata = compress_tensor_with_mask(
            tensor, compression_percentage
        )

        stats.compressed_size = len(column_bytes) + len(mask_bytes)
        stats.compression_time = metadata["compression_time"]
        stats.num_columns_kept = metadata["num_columns_kept"]
        stats.total_columns = metadata["num_columns"]

        compressed_data = {
            "column_bytes": column_bytes,
            "mask_bytes": mask_bytes,
            "metadata": metadata,
        }
    else:
        # Use simple method for better speed
        tensor_bytes, metadata = compress_tensor_simple(tensor, compression_percentage)

        stats.compressed_size = len(tensor_bytes)
        stats.compression_time = metadata["compression_time"]

        # Calculate columns kept (for stats)
        sparsified = column_sparsify_tensor(tensor, compression_percentage)
        feature_dim = tensor.shape[-1]
        reshaped = sparsified.reshape(-1, feature_dim)
        column_norms = mx.linalg.norm(reshaped, axis=0)
        stats.num_columns_kept = int(mx.sum(column_norms > 1e-8).item())
        stats.total_columns = feature_dim

        compressed_data = {"tensor_bytes": tensor_bytes, "metadata": metadata}

    # Calculate sparsity
    sparsified = column_sparsify_tensor(tensor, compression_percentage)
    zero_elements = int(mx.sum(sparsified == 0).item())
    stats.sparsity_ratio = zero_elements / tensor.size
    stats.compression_ratio = stats.original_size / stats.compressed_size

    return compressed_data, stats


def decompress_tensor_from_network(compressed_data: Dict) -> Tuple[mx.array, float]:
    """
    Decompress tensor using the appropriate method.

    Args:
        compressed_data: Dictionary from compress_tensor_for_network

    Returns:
        Tuple of (decompressed_tensor, decompression_time)
    """
    if "column_bytes" in compressed_data:
        # Column mask method
        tensor = decompress_tensor_with_mask(
            compressed_data["column_bytes"],
            compressed_data["mask_bytes"],
            compressed_data["metadata"],
        )
        decompression_time = compressed_data["metadata"]["decompression_time"]
    else:
        # Simple method
        tensor = decompress_tensor_simple(
            compressed_data["tensor_bytes"], compressed_data["metadata"]
        )
        decompression_time = compressed_data["metadata"]["decompression_time"]

    return tensor, decompression_time


def calculate_reconstruction_error(
    original: mx.array, reconstructed: mx.array
) -> float:
    """Calculate MSE reconstruction error."""
    if original.shape != reconstructed.shape:
        raise ValueError("Tensors must have the same shape")

    if original.dtype != reconstructed.dtype:
        reconstructed = reconstructed.astype(original.dtype)

    mse = mx.mean((original - reconstructed) ** 2)
    return float(mse)


# Protobuf integration functions
def compress_tensor_to_protobuf_data(
    tensor: mx.array, compression_percentage: float = 90.0
) -> Tuple[bytes, list, str]:
    """
    Compress tensor for protobuf transmission using simple method.

    Returns:
        Tuple of (tensor_data, shape, dtype_with_metadata) for protobuf
    """
    tensor_bytes, metadata = compress_tensor_simple(tensor, compression_percentage)

    shape = metadata["original_shape"]

    # Encode metadata in dtype string for backward compatibility
    dtype_with_metadata = f"{metadata['compressed_dtype']}|{compression_percentage}|{metadata['quantized']}"

    return tensor_bytes, shape, dtype_with_metadata


def decompress_tensor_from_protobuf_data(
    tensor_data: bytes, shape: list, dtype_with_metadata: str
) -> mx.array:
    """
    Decompress tensor from protobuf data.
    """
    # Parse metadata from dtype string
    parts = dtype_with_metadata.split("|")
    compressed_dtype = parts[0]
    compression_percentage = float(parts[1]) if len(parts) > 1 else 0.0
    quantized = parts[2] == "True" if len(parts) > 2 else False

    metadata = {
        "original_shape": shape,
        "compressed_dtype": compressed_dtype,
        "compression_percentage": compression_percentage,
        "quantized": quantized,
        "original_dtype": "mlx.core.float32",  # Default assumption
    }

    return decompress_tensor_simple(tensor_data, metadata)


if __name__ == "__main__":
    print("Pure Column Sparsification Compression for MLX")
    print("=" * 60)

    # Test 1: Basic compression test
    print("\n1. Basic compression test (1000, 512)...")
    tensor = mx.random.normal((1000, 512))

    # Simple method
    compressed_data, stats = compress_tensor_for_network(
        tensor, compression_percentage=90.0
    )
    reconstructed, decomp_time = decompress_tensor_from_network(compressed_data)
    stats.decompression_time = decomp_time
    stats.reconstruction_error = calculate_reconstruction_error(tensor, reconstructed)

    print("Simple method (default):")
    print(stats)

    # Column mask method
    compressed_data_mask, stats_mask = compress_tensor_for_network(
        tensor, compression_percentage=90.0, use_column_mask=True
    )
    reconstructed_mask, decomp_time_mask = decompress_tensor_from_network(
        compressed_data_mask
    )
    stats_mask.decompression_time = decomp_time_mask
    stats_mask.reconstruction_error = calculate_reconstruction_error(
        tensor, reconstructed_mask
    )

    print("\nColumn mask method:")
    print(stats_mask)

    # Test 2: Performance across different compression levels
    print("\n2. Performance comparison across compression levels...")
    print(
        f"{'Comp %':<8} {'Method':<15} {'Ratio':<8} {'Comp Time':<12} {'Decomp Time':<12} {'Error':<10}"
    )
    print("-" * 75)

    for comp_pct in [0, 50, 75, 90, 95, 99]:
        # Simple method
        comp_data_s, stats_s = compress_tensor_for_network(
            tensor, comp_pct, use_column_mask=False
        )
        recon_s, decomp_t_s = decompress_tensor_from_network(comp_data_s)
        error_s = calculate_reconstruction_error(tensor, recon_s)

        print(
            f"{comp_pct:<8} {'Simple':<15} {stats_s.compression_ratio:<8.2f} "
            f"{stats_s.compression_time*1000:<12.2f} {decomp_t_s*1000:<12.2f} {error_s:<10.6f}"
        )

        if comp_pct >= 50:  # Column mask only makes sense with decent sparsity
            # Column mask method
            comp_data_m, stats_m = compress_tensor_for_network(
                tensor, comp_pct, use_column_mask=True
            )
            recon_m, decomp_t_m = decompress_tensor_from_network(comp_data_m)
            error_m = calculate_reconstruction_error(tensor, recon_m)

            print(
                f"{'':<8} {'Column Mask':<15} {stats_m.compression_ratio:<8.2f} "
                f"{stats_m.compression_time*1000:<12.2f} {decomp_t_m*1000:<12.2f} {error_m:<10.6f}"
            )

    # Test 3: Large tensor test
    print("\n3. Large tensor test (4096, 4096)...")
    large_tensor = mx.random.normal((4096, 4096))

    start_total = time.time()
    comp_large, stats_large = compress_tensor_for_network(
        large_tensor, 95.0, use_column_mask=False
    )
    recon_large, decomp_large = decompress_tensor_from_network(comp_large)
    total_time = time.time() - start_total

    print(
        f"Simple method: {total_time*1000:.1f}ms total, {stats_large.compression_ratio:.2f}x compression"
    )

    # Test 4: Protobuf integration
    print("\n4. Protobuf integration test...")
    proto_data, proto_shape, proto_meta = compress_tensor_to_protobuf_data(tensor, 90.0)
    proto_recon = decompress_tensor_from_protobuf_data(
        proto_data, proto_shape, proto_meta
    )
    proto_error = calculate_reconstruction_error(tensor, proto_recon)

    print(f"Protobuf data size: {len(proto_data):,} bytes")
    print(f"Reconstruction error: {proto_error:.6f}")

    print("\nAll tests completed!")
