"""Tests: utils.model I/O helpers that touch the filesystem and mmap."""

import json
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from dnet.utils.model import (
    get_model_config_json,
    resolve_tokenizer_dir,
    MappedFile,
    TensorInfo,
    load_weight,
)


pytestmark = [pytest.mark.core]


def test_get_model_config_json_local(tmp_path):
    cfg = {"model_type": "demo", "num_hidden_layers": 3}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    out = get_model_config_json(str(tmp_path))
    assert out["model_type"] == "demo" and out["num_hidden_layers"] == 3


def test_resolve_tokenizer_dir_local(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")  # create a minimal tokenizer asset
    p = resolve_tokenizer_dir(str(tmp_path))
    assert str(p) == str(tmp_path)


def test_mapped_file_opens_and_maps(tmp_path):
    data = b"abcdef"
    fpath = tmp_path / "x.bin"
    fpath.write_bytes(data)
    mf = MappedFile(str(fpath))
    try:
        assert mf.mmap.size() == len(data)
        assert mf.base_addr > 0
    finally:
        mf.mmap.close()
        mf.file.close()


def test_load_weight_float32_reads_bytes(tmp_path):
    arr = np.array([1.0, -2.5], dtype=np.float32)
    fpath = tmp_path / "w.bin"
    fpath.write_bytes(arr.tobytes())
    ti = TensorInfo(
        dtype="F32", shape=(2,), size_bytes=arr.nbytes, offset=0, filename=str(fpath)
    )
    mapped = {}
    out = load_weight(ti, mapped)
    assert tuple(out.shape) == (2,)
    got = np.array(out.tolist(), dtype=np.float32)
    assert np.allclose(got, arr)


def test_load_weight_bf16_reads_bytes(tmp_path):
    # bf16 payload: values 1.0 and 0.5 -> upper 16 bits 0x3F80, 0x3F00
    bf16_words = np.array([0x3F80, 0x3F00], dtype=np.uint16)
    fpath = tmp_path / "wb.bin"
    fpath.write_bytes(bf16_words.tobytes())
    ti = TensorInfo(
        dtype="BF16",
        shape=(2,),
        size_bytes=bf16_words.nbytes,
        offset=0,
        filename=str(fpath),
    )
    mapped = {}
    out = load_weight(ti, mapped)
    assert tuple(out.shape) == (2,)
    out32 = out.astype(mx.float32)  # cast back to float32 and compare approximately
    got = np.array(out32.tolist(), dtype=np.float32)
    assert np.allclose(got, np.array([1.0, 0.5], dtype=np.float32), atol=1e-3)
