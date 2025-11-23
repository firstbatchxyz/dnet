"""Tests: tensor helpers to_bytes and bytes_to_tensor roundtrips/casting."""

import pytest
import numpy as np

mx = pytest.importorskip("mlx.core")

from dnet.core.tensor import to_bytes
from dnet.utils.serialization import bytes_to_tensor

pytestmark = [pytest.mark.core]


def test_to_bytes_numpy_int32_size():
    arr = np.array([1, 2, 3], dtype=np.int32)
    b = to_bytes(arr, wire_dtype_str="int32", wire_mx_dtype=mx.int32)
    assert isinstance(b, (bytes, bytearray))
    assert len(b) == arr.size * 4


def test_to_bytes_mlx_int32_size():
    arr = mx.array([1, 2, 3], dtype=mx.int32)
    b = to_bytes(arr, wire_dtype_str="int32", wire_mx_dtype=mx.int32)
    assert isinstance(b, (bytes, bytearray))
    assert len(b) == 3 * 4


def test_bytes_to_tensor_roundtrip_float32():
    arr = mx.array([1.5, -2.0, 0.0], dtype=mx.float32)
    b = to_bytes(arr, wire_dtype_str="float32", wire_mx_dtype=mx.float32)
    out = bytes_to_tensor(b, "float32")
    assert str(out.dtype) == str(mx.float32)
    assert out.shape[0] == 3
