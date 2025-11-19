import numpy as np
import mlx.core as mx
import time
from typing import Optional, Tuple
from dnet.utils.serialization import dtype_map, mlx_dtype_map, tensor_to_bytes
from dnet.compression import decompress_tensor_from_protobuf_data
from dnet.core.types.messages import ActivationMessage
from dnet.protos.dnet_ring_pb2 import ActivationRequest
from dnet.utils.logger import logger


class ActivationCodec:
    """
    Handles the translation between Network Protobufs (ActivationRequest)
    and Runtime Objects (ActivationMessage + Shared Memory Pools).
    """

    def __init__(self, runtime):
        self.runtime = runtime

    def deserialize(self, request: ActivationRequest) -> Optional[ActivationMessage]:
        """
        Parses an incoming Proto, allocates memory in the pool,
        decompresses data, and returns a ready-to-compute ActivationMessage.
        """
        if self.runtime.input_pool is None:
            logger.error("Shard %s: input pool not initialized", self.runtime.shard_id)
            return None

        activation = request.activation
        pool_id = None

        try:
            # Compressed Tensors
            if "|" in activation.dtype:
                deq = decompress_tensor_from_protobuf_data(
                    tensor_data=activation.data,
                    shape=list(activation.shape),
                    dtype_with_metadata=activation.dtype,
                )
                pool_id = self.runtime.input_pool.allocate_for_layer(
                    layer_id=activation.layer_id,
                    dtype=deq.dtype,
                    shape=tuple(deq.shape),
                )
                if pool_id is not None:
                    buffer = self.runtime.input_pool.get_buffer(pool_id)
                    flat = deq.reshape(-1)
                    buffer[: flat.size] = flat
                    msg = ActivationMessage.from_proto(request, pool_id)
                    msg.dtype = str(deq.dtype)
                    msg.shape = tuple(deq.shape)
                    return msg

            # Tokens (Integer sequences)
            elif activation.dtype == "tokens":
                tokens = np.frombuffer(activation.data, dtype=np.int32)
                shp = (int(len(tokens)),)
                pool_id = self.runtime.input_pool.allocate_for_layer(
                    layer_id=activation.layer_id, dtype=mx.int32, shape=shp
                )
                if pool_id is not None:
                    buffer = self.runtime.input_pool.get_buffer(pool_id)
                    buffer[: len(tokens)] = tokens
                    msg = ActivationMessage.from_proto(request, pool_id)
                    msg.dtype = "tokens"
                    msg.shape = shp
                    return msg

            # Standard Raw Tensors
            else:
                # Validation
                expected = int(np.prod(activation.shape)) * np.dtype(dtype_map[activation.dtype]).itemsize
                actual = len(activation.data)
                if expected != actual:
                    logger.error("Payload mismatch nonce=%s: exp=%d act=%d", request.nonce, expected, actual)
                    return None

                pool_id = self.runtime.input_pool.allocate_for_layer(
                    layer_id=activation.layer_id,
                    dtype=mlx_dtype_map[activation.dtype],
                    shape=tuple(activation.shape),
                )
                if pool_id is not None:
                    buffer = self.runtime.input_pool.get_buffer(pool_id)
                    input_data = np.frombuffer(activation.data, dtype=dtype_map[activation.dtype])
                    buffer[: len(input_data)] = input_data
                    return ActivationMessage.from_proto(request, pool_id)

        except Exception as e:
            logger.error(f"Deserialization error for nonce {request.nonce}: {e}")
            # Cleanup if allocation happened but fill failed
            if pool_id is not None:
                self.runtime.input_pool.release(pool_id)
            return None

        return None

    def serialize(self, msg: ActivationMessage, transport_config) -> bytes:
        """
        Reads from output pool/tensor, compresses, and returns bytes + wire dtype.
        """
        shaped = msg.tensor
        if shaped is None:
            if self.runtime.output_pool is None:
                raise ValueError("No output pool and no tensor to serialize")
            output_buffer = self.runtime.output_pool.get_buffer(msg.pool_id)
            data_size = int(np.prod(msg.shape))
            shaped = output_buffer[:data_size].reshape(msg.shape)

        data = self.to_bytes(
            shaped,
            wire_dtype_str=self.runtime._wire_dtype_str,
            wire_mx_dtype=self.runtime._wire_mx_dtype,
            compress=transport_config.compress,
            compress_min_bytes=transport_config.compress_min_bytes,
        )

        # Clean up reference immediately to assist GC
        msg.tensor = None
        return data

    @staticmethod
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
            Tuple of (raw_bytes, ser_ms, cast_ms)
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