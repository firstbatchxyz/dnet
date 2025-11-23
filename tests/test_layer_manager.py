"""Tests: LayerManager file I/O, load to GPU, release, and config flags."""

import os
import pytest
import numpy as np

mx = pytest.importorskip("mlx.core")

from dnet.utils.layer_manager import LayerManager
from dnet.utils.model import ModelMetadata, TensorInfo

pytestmark = [pytest.mark.core]


def _create_model_meta(tmpdir, layers, tensor_bytes, dtype="F32"):
    fpath = os.path.join(tmpdir, "w.bin")
    with open(fpath, "wb") as f:
        f.write(tensor_bytes)
    weight_info = {}
    for li in layers:
        weight_info[li] = {
            "weight": TensorInfo(
                dtype=dtype,
                shape=(len(tensor_bytes) // (2 if dtype == "BF16" else 4),),
                size_bytes=len(tensor_bytes),
                offset=0,
                filename=fpath,
            )
        }
    meta = ModelMetadata(
        path=tmpdir,
        weight_info=weight_info,
        embed_tokens={},
        lm_head={},
        norm={},
        config={"model_type": "test", "num_hidden_layers": max(layers) + 1},
    )
    return meta, fpath


def test_init_fastpath_and_off(tmp_path):
    arr = np.array([1.0, 2.0], dtype=np.float32).tobytes()
    meta, _ = _create_model_meta(str(tmp_path), [0], arr)
    lm = LayerManager(
        meta, [0], thread_pool_size=1, use_mxload_fastpath=True, prefetch_mode="off"
    )
    assert lm._use_mxload_fastpath is True
    assert lm._prefetch_mode == "off"
    assert isinstance(lm.assigned_layers, set) and 0 in lm.assigned_layers


def test_load_layer_to_gpu_reads_bytes(tmp_path):
    vals = np.array([3.0, -5.5, 7.25], dtype=np.float32)
    meta, _ = _create_model_meta(str(tmp_path), [1], vals.tobytes())
    lm = LayerManager(
        meta, [1], thread_pool_size=1, use_mxload_fastpath=False, prefetch_mode="off"
    )
    out = lm.load_layer_to_gpu(1)
    key = "layers.1.weight"
    assert key in out
    got = np.array(out[key].tolist(), dtype=np.float32)
    assert np.allclose(got, vals)


def test_load_layer_to_gpu_unassigned_raises(tmp_path):
    vals = np.array([1.0], dtype=np.float32)
    meta, _ = _create_model_meta(str(tmp_path), [0], vals.tobytes())
    lm = LayerManager(
        meta, [0], thread_pool_size=1, use_mxload_fastpath=False, prefetch_mode="off"
    )
    with pytest.raises(RuntimeError):
        _ = lm.load_layer_to_gpu(99)


def test_prefetch_off_returns_true(tmp_path):
    vals = np.array([1.0], dtype=np.float32)
    meta, _ = _create_model_meta(str(tmp_path), [0], vals.tobytes())
    lm = LayerManager(
        meta, [0], thread_pool_size=1, use_mxload_fastpath=False, prefetch_mode="off"
    )
    assert lm.prefetch_layer(0) is True


def test_prefetch_fastpath_returns_true_and_false_when_missing(tmp_path):
    vals = np.array([1.0], dtype=np.float32)
    meta, fpath = _create_model_meta(str(tmp_path), [2], vals.tobytes())
    lm = LayerManager(
        meta, [2], thread_pool_size=1, use_mxload_fastpath=True, prefetch_mode="off"
    )
    assert lm.prefetch_layer(2) is True
    assert lm.prefetch_layer(12345) is False  # missing layer index


def test_prefetch_modes_sequential_and_full_success(tmp_path, monkeypatch):
    vals = np.array([1.0, 2.0], dtype=np.float32)
    meta, _ = _create_model_meta(str(tmp_path), [0], vals.tobytes())
    lm = LayerManager(
        meta,
        [0],
        thread_pool_size=1,
        use_mxload_fastpath=False,
        prefetch_mode="sequential",
    )
    monkeypatch.setattr(
        "dnet.utils.layer_manager.libc.madvise", lambda *a: 0, raising=True
    )
    assert lm.prefetch_layer(0) is True
    lm2 = LayerManager(
        meta, [0], thread_pool_size=1, use_mxload_fastpath=False, prefetch_mode="full"
    )
    assert lm2.prefetch_layer(0) is True


def test_prefetch_mode_full_failure(tmp_path, monkeypatch):
    vals = np.array([1.0], dtype=np.float32)
    meta, _ = _create_model_meta(str(tmp_path), [0], vals.tobytes())
    lm = LayerManager(
        meta, [0], thread_pool_size=1, use_mxload_fastpath=False, prefetch_mode="full"
    )
    monkeypatch.setattr(
        "dnet.utils.layer_manager.libc.madvise", lambda *a: 1, raising=True
    )
    assert lm.prefetch_layer(0) is False


def test_release_layer_nonfastpath_true_and_false(tmp_path, monkeypatch):
    vals = np.array([1.0], dtype=np.float32)
    meta, _ = _create_model_meta(str(tmp_path), [1], vals.tobytes())
    lm = LayerManager(
        meta, [1], thread_pool_size=1, use_mxload_fastpath=False, prefetch_mode="off"
    )
    monkeypatch.setattr(
        "dnet.utils.layer_manager.libc.madvise", lambda *a: 0, raising=True
    )
    assert lm.release_layer(1) is True
    monkeypatch.setattr(
        "dnet.utils.layer_manager.libc.madvise", lambda *a: 1, raising=True
    )
    assert lm.release_layer(1) is False


def test_fastpath_load_with_prefix_keys(tmp_path, monkeypatch):
    vals = np.array([4.0, 5.0], dtype=np.float32)
    meta, f = _create_model_meta(str(tmp_path), [3], vals.tobytes())
    lm = LayerManager(
        meta, [3], thread_pool_size=1, use_mxload_fastpath=True, prefetch_mode="off"
    )

    def fake_load(path):
        return {
            "model.layers.3.weight": mx.array(vals),
            "layers.3.proj": mx.array(vals),
        }

    monkeypatch.setattr("mlx.core.load", fake_load, raising=True)
    out = lm.load_layer_to_gpu(3)
    assert "layers.3.weight" in out and "layers.3.proj" in out


def test_fastpath_load_fallback_on_nondict(tmp_path, monkeypatch):
    vals = np.array([7.0], dtype=np.float32)
    meta, f = _create_model_meta(str(tmp_path), [4], vals.tobytes())
    lm = LayerManager(
        meta, [4], thread_pool_size=1, use_mxload_fastpath=True, prefetch_mode="off"
    )
    monkeypatch.setattr("mlx.core.load", lambda p: mx.array(vals), raising=True)
    out = lm.load_layer_to_gpu(4)
    assert "layers.4.weight" in out


def test_bf16_decode_path(tmp_path):
    u16 = np.array([0x3F80, 0x4000], dtype=np.uint16).tobytes()
    meta, _ = _create_model_meta(str(tmp_path), [0], u16, dtype="BF16")
    lm = LayerManager(
        meta, [0], thread_pool_size=1, use_mxload_fastpath=False, prefetch_mode="off"
    )
    out = lm.load_layer_to_gpu(0)
    arr = out["layers.0.weight"]
    assert str(arr.dtype) == str(mx.bfloat16)


def test_async_prefetch_returns_future(tmp_path):
    vals = np.array([1.0], dtype=np.float32)
    meta, _ = _create_model_meta(str(tmp_path), [0], vals.tobytes())
    lm = LayerManager(
        meta, [0], thread_pool_size=1, use_mxload_fastpath=False, prefetch_mode="off"
    )
    fut = lm.async_prefetch(0)
    assert hasattr(fut, "result")
    assert fut.result(timeout=2) is True


def test_release_layer_fastpath_true(tmp_path):
    vals = np.array([1.0], dtype=np.float32)
    meta, _ = _create_model_meta(str(tmp_path), [0], vals.tobytes())
    lm = LayerManager(
        meta, [0], thread_pool_size=1, use_mxload_fastpath=True, prefetch_mode="off"
    )
    assert lm.release_layer(0) is True


def test_memadvise_layer_nonassigned_returns_false(tmp_path):
    vals = np.array([1.0], dtype=np.float32)
    meta, _ = _create_model_meta(str(tmp_path), [0], vals.tobytes())
    lm = LayerManager(
        meta, [0], thread_pool_size=1, use_mxload_fastpath=False, prefetch_mode="off"
    )
    assert lm._memadvise_layer(99, 3) is False
