"""Tests: ActivationCodec serialize/deserialize for tokens/raw/compressed and errors."""

# ruff: noqa: E402
import pytest
import numpy as np

mx = pytest.importorskip("mlx.core")

from dnet.protos.dnet_ring_pb2 import Activation, ActivationRequest
from dnet.shard.codec import ActivationCodec
from tests.fakes import FakeRuntimeMinimal

pytestmark = [pytest.mark.shard, pytest.mark.codec]


def _create_codec():
    rt = FakeRuntimeMinimal()
    return ActivationCodec(rt), rt


def test_deserialize_tokens_success():
    codec, rt = _create_codec()
    toks = np.array([1, 2, 3], dtype=np.int32)
    act = Activation(
        data=toks.tobytes(), batch_size=3, shape=[len(toks)], dtype="tokens", layer_id=0
    )
    req = ActivationRequest(
        nonce="n1", activation=act, timestamp=0, node_origin="S", callback_url="cb"
    )
    msg = codec.deserialize(req)
    assert msg is not None and msg.dtype == "tokens" and tuple(msg.shape) == (3,)


def test_deserialize_raw_tensor_success():
    codec, rt = _create_codec()
    arr = np.array([1.5, -2.0], dtype=np.float32)
    act = Activation(
        data=arr.tobytes(), batch_size=1, shape=[2], dtype="float32", layer_id=1
    )
    req = ActivationRequest(
        nonce="n2", activation=act, timestamp=0, node_origin="S", callback_url="cb"
    )
    msg = codec.deserialize(req)
    assert msg is not None and tuple(msg.shape) == (2,)


def test_deserialize_mismatch_returns_none():
    codec, rt = _create_codec()
    arr = np.array([1.0], dtype=np.float32)
    act = Activation(  # shape=2, but only one float provided
        data=arr.tobytes(), batch_size=1, shape=[2], dtype="float32", layer_id=1
    )
    req = ActivationRequest(
        nonce="n3", activation=act, timestamp=0, node_origin="S", callback_url="cb"
    )
    assert codec.deserialize(req) is None


def test_deserialize_compressed_uses_decompress(monkeypatch):
    codec, rt = _create_codec()
    out = mx.array([1.0, 2.0], dtype=mx.float16)

    def _decomp(tensor_data, shape, dtype_with_metadata):
        return out

    monkeypatch.setattr(
        "dnet.shard.codec.decompress_tensor_from_protobuf_data", _decomp, raising=True
    )
    act = Activation(
        data=b"xx", batch_size=1, shape=[2], dtype="float16|lz4", layer_id=0
    )
    req = ActivationRequest(
        nonce="n4", activation=act, timestamp=0, node_origin="S", callback_url="cb"
    )
    msg = codec.deserialize(req)
    assert msg is not None and msg.dtype == str(out.dtype)


def test_serialize_with_tensor_and_wire_dtype():
    codec, rt = _create_codec()
    from dnet.core.types.messages import ActivationMessage

    t = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    msg = ActivationMessage(
        nonce="n",
        pool_id=0,
        batch_size=1,
        shape=(3,),
        dtype="float32",
        layer_id=0,
        timestamp=0,
        node_origin="S",
        callback_url="cb",
        tensor=t,
    )
    from dnet.shard.config import TransportConfig

    data, meta = codec.serialize(
        msg,
        transport_config=TransportConfig(
            compress=False, compress_min_bytes=65536, wire_mode="fp16"
        ),
    )
    # float16 wire dtype -> 2 bytes per element
    assert isinstance(data, (bytes, bytearray)) and len(data) == 3 * 2
    assert meta == rt._wire_dtype_str


def test_serialize_reads_from_output_pool_when_no_tensor():
    codec, rt = _create_codec()
    from dnet.core.types.messages import ActivationMessage
    from dnet.shard.config import TransportConfig

    pool_id = rt.output_pool.allocate_for_layer(
        layer_id=0, dtype=mx.float32, shape=(3,)
    )
    buf = rt.output_pool.get_buffer(pool_id)
    buf[:3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # message with no tensor should read from output pool by pool_id
    msg = ActivationMessage(
        nonce="n",
        pool_id=pool_id,
        batch_size=1,
        shape=(3,),
        dtype="float32",
        layer_id=0,
        timestamp=0,
        node_origin="S",
        callback_url="cb",
        tensor=None,
    )
    data, meta = codec.serialize(
        msg,
        transport_config=TransportConfig(
            compress=False, compress_min_bytes=0, wire_mode="fp16"
        ),
    )
    assert isinstance(data, (bytes, bytearray)) and len(data) == 3 * 2
    assert meta == rt._wire_dtype_str


def test_deserialize_returns_none_when_no_input_pool():
    codec, rt = _create_codec()
    codec.runtime.input_pool = None  # disable input pool to trigger early None
    act = Activation(data=b"", batch_size=1, shape=[0], dtype="float32", layer_id=0)
    req = ActivationRequest(
        nonce="n5", activation=act, timestamp=0, node_origin="S", callback_url="cb"
    )
    assert codec.deserialize(req) is None


def test_deserialize_allocation_failure_returns_none(monkeypatch):
    codec, rt = _create_codec()

    def _fail_alloc(*a, **k):
        return None

    monkeypatch.setattr(rt.input_pool, "allocate_for_layer", _fail_alloc, raising=True)
    toks = np.array([1, 2], dtype=np.int32)
    act = Activation(
        data=toks.tobytes(), batch_size=2, shape=[len(toks)], dtype="tokens", layer_id=0
    )
    req = ActivationRequest(
        nonce="n6", activation=act, timestamp=0, node_origin="S", callback_url="cb"
    )
    assert codec.deserialize(req) is None


def test_deserialize_compressed_with_bare_pipe_invokes_decompress(monkeypatch):
    codec, rt = _create_codec()
    out = mx.array([3.0, 4.0], dtype=mx.float16)
    called = {"ok": False, "args": None}

    def _decomp(tensor_data, shape, dtype_with_metadata):
        called["ok"] = True
        called["args"] = (tensor_data, tuple(shape), dtype_with_metadata)
        return out

    monkeypatch.setattr(
        "dnet.shard.codec.decompress_tensor_from_protobuf_data", _decomp, raising=True
    )
    act = Activation(data=b"xx", batch_size=1, shape=[2], dtype="|", layer_id=0)
    req = ActivationRequest(
        nonce="n7", activation=act, timestamp=0, node_origin="S", callback_url="cb"
    )
    msg = codec.deserialize(req)
    assert called["ok"] is True and msg is not None
    assert msg.dtype == str(out.dtype) and tuple(msg.shape) == tuple(out.shape)


def test_deserialize_compressed_decompress_error_returns_none(monkeypatch):
    codec, rt = _create_codec()

    def _boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(
        "dnet.shard.codec.decompress_tensor_from_protobuf_data", _boom, raising=True
    )
    act = Activation(
        data=b"bad", batch_size=1, shape=[2], dtype="float16|lz4", layer_id=0
    )
    req = ActivationRequest(
        nonce="n8", activation=act, timestamp=0, node_origin="S", callback_url="cb"
    )
    assert codec.deserialize(req) is None


def test_serialize_deserialize_q8_dense():
    codec, rt = _create_codec()
    from dnet.core.types.messages import ActivationMessage
    from dnet.shard.config import TransportConfig

    # Create 3D tensor to test flattening/restoration
    B, L, D = 2, 4, 8
    original_shape = (B, L, D)
    # Use random data
    original_data = np.random.randn(*original_shape).astype(np.float16)
    t = mx.array(original_data)

    msg = ActivationMessage(
        nonce="q8",
        pool_id=0,
        batch_size=B,
        shape=original_shape,
        dtype="float16",
        layer_id=0,
        timestamp=0,
        node_origin="S",
        callback_url="cb",
        tensor=t,
    )

    # Serialize with q8_dense
    data, meta = codec.serialize(
        msg,
        transport_config=TransportConfig(wire_mode="q8_dense"),
    )

    # Check metadata
    assert "fmt=q8_dense_v0" in meta
    assert f"rows={B * L}" in meta
    assert f"cols={D}" in meta

    # Mock input pool allocation for deserialization
    # We need to mock allocate_for_layer and get_buffer
    pool_id = 123

    def _alloc(layer_id, dtype, shape):
        return pool_id

    # Use a real numpy array as the buffer backing
    buffer_backing = np.zeros(B * L * D, dtype=np.float16)

    def _get_buf(pid):
        if pid == pool_id:
            return buffer_backing
        return None

    codec.runtime.input_pool.allocate_for_layer = _alloc
    codec.runtime.input_pool.get_buffer = _get_buf

    # Deserialize
    act = Activation(
        data=data,
        batch_size=B,
        shape=list(original_shape),
        dtype=meta,
        layer_id=0,
    )
    req = ActivationRequest(
        nonce="q8", activation=act, timestamp=0, node_origin="S", callback_url="cb"
    )

    decoded_msg = codec.deserialize(req)

    assert decoded_msg is not None
    assert decoded_msg.pool_id == pool_id
    assert tuple(decoded_msg.shape) == original_shape

    # Check data accuracy
    reconstructed_data = buffer_backing.reshape(original_shape)
    diff = np.abs(original_data - reconstructed_data)
    max_diff = np.max(diff)

    # Q8 quantization error should be small (< 0.1 for this range)
    assert max_diff < 0.1
