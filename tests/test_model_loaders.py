"""Tests: model loader helpers reading real bytes (embeddings, norm, lm_head)."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from dnet.utils.model import TensorInfo, load_embeddings, load_final_norm, load_lm_head
from tests.fakes import FakeModelMetadata, FakeRingModel

pytestmark = [pytest.mark.core]


def _create_file(tmp_path, arr: np.ndarray):
    p = tmp_path / "w.bin"
    with open(p, "wb") as f:
        f.write(arr.tobytes())
    return str(p)


def test_load_embeddings_reads_bytes(tmp_path):
    vals = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    fpath = _create_file(tmp_path, vals)
    md = FakeModelMetadata(
        weight_info={}, model_type="demo", model_config={}, num_layers=1
    )
    md.embed_tokens["weight"] = TensorInfo(
        dtype="F32",
        shape=list(vals.shape),
        size_bytes=vals.nbytes,
        offset=0,
        filename=fpath,
    )
    model = FakeRingModel()
    n = load_embeddings(md, model)
    assert n == 1
    assert "embed_tokens.weight" in model.loaded
    got = np.array(model.loaded["embed_tokens.weight"].tolist(), dtype=np.float32)
    assert np.allclose(got, vals)


def test_load_final_norm_reads_bytes(tmp_path):
    vals = np.array([0.25, 0.5], dtype=np.float32)
    fpath = _create_file(tmp_path, vals)
    md = FakeModelMetadata(
        weight_info={}, model_type="demo", model_config={}, num_layers=1
    )
    md.norm["weight"] = TensorInfo(
        dtype="F32",
        shape=list(vals.shape),
        size_bytes=vals.nbytes,
        offset=0,
        filename=fpath,
    )
    model = FakeRingModel()
    n = load_final_norm(md, model)
    assert n == 1 and "norm.weight" in model.loaded


def test_load_lm_head_dense_shapes(tmp_path):
    hidden = 2
    vocab = 3
    vals = np.arange(hidden * vocab, dtype=np.float32).reshape(hidden, vocab)
    fpath = _create_file(tmp_path, vals)
    md = FakeModelMetadata(
        weight_info={}, model_type="demo", model_config={}, num_layers=1
    )
    md.lm_head["weight"] = TensorInfo(
        dtype="F32",
        shape=[hidden, vocab],
        size_bytes=vals.nbytes,
        offset=0,
        filename=fpath,
    )
    model = FakeRingModel(hidden_size=hidden, vocab_size=vocab)
    n = load_lm_head(md, model)
    assert n == 1 and "lm_head.weight" in model.loaded
