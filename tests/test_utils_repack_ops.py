"""Tests: utils.repack functions that touch filesystem and env.

Covers:
- _sanitize_model_id and _hash_layers
- _copy_non_weight_artifacts (copies non-weight files)
- delete_repacked_layers for all code paths
- ensure_repacked_for_layers manifest behavior with DNET_REPACK_DIR
"""

import json
import pytest

from dnet.utils.repack import (
    _sanitize_model_id,
    _hash_layers,
    _copy_non_weight_artifacts,
    delete_repacked_layers,
    ensure_repacked_for_layers,
)

pytestmark = [pytest.mark.core]


# multiple special characters collapse to underscores; trailing special removed
def test_sanitize_and_hash_layers_basic():
    assert _sanitize_model_id("hf://a/b c?") == "hf___a_b_c"
    assert _hash_layers([3, 1, 2]) == _hash_layers([2, 3, 1])


def test_copy_non_weight_artifacts(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "tokenizer.json").write_text("{}")  # non-weight artifacts
    (src / "config.json").write_text("{}")
    (src / "note.txt").write_text("ok")
    (src / "a.safetensors").write_bytes(b"x")  # weight-like files that must be skipped
    (src / "b.bin").write_bytes(b"x")
    _copy_non_weight_artifacts(src, dst)
    names = sorted(p.name for p in dst.iterdir())
    assert names == ["config.json", "note.txt", "tokenizer.json"]


def test_delete_repacked_layers_all_flag(tmp_path, monkeypatch):
    base = tmp_path / "repacked"
    target = base / "m"
    target.mkdir(parents=True)
    monkeypatch.setenv("DNET_REPACK_DIR", str(base))
    removed = delete_repacked_layers(all_flag=True)
    assert str(base) in removed and not base.exists()


def test_delete_repacked_layers_model_id_bucket(tmp_path, monkeypatch):
    base = tmp_path / "repacked"
    target = base / "hf___m"
    target.mkdir(parents=True)
    monkeypatch.setenv("DNET_REPACK_DIR", str(base))
    removed = delete_repacked_layers(model_id="hf://m")
    assert str(target) in removed and not target.exists()


def test_delete_repacked_layers_current_path_manifest_case(tmp_path, monkeypatch):
    base = tmp_path / "repacked"
    bucket = base / "hf___z"
    bucket.mkdir(parents=True)
    pcur = tmp_path / "any_where"  # current path with a manifest pointing to model_id
    pcur.mkdir()
    (pcur / "repack-manifest.json").write_text(json.dumps({"model_id": "hf://z"}))
    monkeypatch.setenv("DNET_REPACK_DIR", str(base))
    removed = delete_repacked_layers(current_model_path=str(pcur))
    assert str(bucket) in removed and not bucket.exists()


def test_delete_repacked_layers_current_path_inside_base(tmp_path, monkeypatch):
    base = tmp_path / "repacked"
    bucket = base / "safeid"
    inner = bucket / "x" / "y"
    inner.mkdir(parents=True)
    monkeypatch.setenv("DNET_REPACK_DIR", str(base))
    removed = delete_repacked_layers(current_model_path=str(inner))
    assert str(bucket) in removed and not bucket.exists()


def test_delete_repacked_layers_current_path_fallback_sanitization(
    tmp_path, monkeypatch
):
    base = tmp_path / "repacked"
    bucket = base / _sanitize_model_id("path/to/special")
    bucket.mkdir(parents=True)
    monkeypatch.setenv("DNET_REPACK_DIR", str(base))
    removed = delete_repacked_layers(current_model_path="path/to/special")
    assert str(bucket) in removed and not bucket.exists()


def test_ensure_repacked_for_layers_creates_and_manifest(tmp_path, monkeypatch):
    # direct repack: monkeypatch repack_per_layer to drop a dummy layer file
    base = tmp_path / "repacked"
    monkeypatch.setenv("DNET_REPACK_DIR", str(base))

    from dnet.utils import repack as _mod

    def _fake_repack(model_path, assigned_layers, out_root):
        out_root.mkdir(parents=True, exist_ok=True)
        first = sorted(set(assigned_layers))[0]
        (out_root / f"layer_{first:04d}.safetensors").write_bytes(b"x")

    monkeypatch.setattr(_mod, "repack_per_layer", _fake_repack, raising=True)

    # inject minimal model metadata for manifest creation
    from tests.fakes import FakeModelMetadata

    monkeypatch.setattr(
        _mod,
        "get_model_metadata",
        lambda _: FakeModelMetadata(num_layers=4, path="/src"),
        raising=True,
    )

    out1, repacked1 = ensure_repacked_for_layers("hf://m", [2, 3])
    assert out1.exists() and repacked1 is True

    # remove manifest to test manifest-writer path on second run
    man = out1 / "repack-manifest.json"
    if man.exists():
        man.unlink()
    out2, repacked2 = ensure_repacked_for_layers("hf://m", [2, 3])
    assert repacked2 is False and man.exists()


def test_ensure_repacked_for_layers_recreates_when_expected_missing(
    tmp_path, monkeypatch
):
    # create a real repack once, then remove the expected first-layer file to force repack
    base = tmp_path / "repacked"
    monkeypatch.setenv("DNET_REPACK_DIR", str(base))

    from dnet.utils import repack as _mod

    def _first_repack(model_path, assigned_layers, out_root):
        out_root.mkdir(parents=True, exist_ok=True)
        first = sorted(set(assigned_layers))[0]
        (out_root / f"layer_{first:04d}.safetensors").write_bytes(b"x")
        (out_root / "repack-manifest.json").write_text(
            json.dumps({"model_id": model_path})
        )

    monkeypatch.setattr(_mod, "repack_per_layer", _first_repack, raising=True)
    out1, repacked1 = ensure_repacked_for_layers("hf://m2", [1, 2])
    assert out1.exists() and repacked1 is True

    expected = out1 / "layer_0001.safetensors"
    if expected.exists():
        expected.unlink()

    called = {"n": 0}

    def _second_repack(model_path, assigned_layers, out_root):
        called["n"] += 1
        (out_root / "layer_0001.safetensors").write_bytes(b"y")

    monkeypatch.setattr(_mod, "repack_per_layer", _second_repack, raising=True)
    out2, repacked2 = ensure_repacked_for_layers("hf://m2", [1, 2])
    assert repacked2 is True and called["n"] == 1
    assert (out2 / "layer_0001.safetensors").exists()
