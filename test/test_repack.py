"""Tests for model weight repacking functionality."""

import os
from pathlib import Path
import tempfile

import mlx.core as mx
import pytest

from dnet.utils.repack import repack_windows, chunk_layers
from dnet.utils.model import get_model_metadata, load_weight


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_model_path():
    """Get path to a test model.

    For CI: Use a tiny test model
    For local: Can override with env var TEST_MODEL_PATH
    """
    if os.environ.get("TEST_MODEL_PATH"):
        return os.environ["TEST_MODEL_PATH"]
    return "Qwen/Qwen3-4B-MLX-4bit"  # Small enough for tests


def test_chunk_layers():
    """Test the chunk_layers function splits lists correctly."""
    layers = [0, 1, 2, 3]

    # Single item per chunk
    chunks = chunk_layers(layers, 1)
    assert chunks == [[0], [1], [2], [3]]

    # Two items per chunk
    chunks = chunk_layers(layers, 2)
    assert chunks == [[0, 1], [2, 3]]

    # Larger than list
    chunks = chunk_layers(layers, 5)
    assert chunks == [[0, 1, 2, 3]]

    # Empty list
    chunks = chunk_layers([], 2)
    assert chunks == []


def test_repack_single_window(test_model_path, temp_cache_dir):
    """Test repacking a single window of layers."""
    # Get model metadata to verify layer count
    md = get_model_metadata(test_model_path)
    layer_indices = [0]  # Test first layer

    # Repack the layers
    out_prefix = temp_cache_dir / "test"
    out_files = repack_windows(Path(md.path), [layer_indices], out_prefix)

    assert len(out_files) == 1
    out_file = out_files[0]

    # Verify output file exists
    assert out_file.exists()
    assert out_file.name == "test_000-000.safetensors"

    # Load and verify the repacked weights
    repacked = mx.load(str(out_file))

    # Check that all expected tensors are present
    for layer_idx in layer_indices:
        for k in md.weight_info[layer_idx].keys():
            tensor_name = f"model.layers.{layer_idx}.{k}"
            assert tensor_name in repacked, f"Missing tensor {tensor_name}"


def test_repack_multiple_windows(test_model_path, temp_cache_dir):
    """Test repacking multiple windows of layers."""
    md = get_model_metadata(test_model_path)
    layer_indices = [0, 1]  # Test first two layers
    chunks = chunk_layers(layer_indices, window_size=1)  # Each layer in own window

    # Repack layers
    out_prefix = temp_cache_dir / "test"
    out_files = repack_windows(Path(md.path), chunks, out_prefix)

    assert len(out_files) == 2  # Should get two files

    # Verify all files exist with correct naming
    for i, layer_idx in enumerate(layer_indices):
        out_file = out_files[i]
        assert out_file.exists()
        assert out_file.name == f"test_{layer_idx:03d}-{layer_idx:03d}.safetensors"

        # Load and verify contents
        repacked = mx.load(str(out_file))
        for k in md.weight_info[layer_idx].keys():
            tensor_name = f"model.layers.{layer_idx}.{k}"
            assert tensor_name in repacked, f"Missing tensor {tensor_name}"


def test_repack_invalid_window(test_model_path, temp_cache_dir):
    """Test repacking with invalid layer indices."""
    md = get_model_metadata(test_model_path)
    invalid_idx = md.num_layers + 1  # Beyond model's layer count

    out_prefix = temp_cache_dir / "test"
    # Should handle invalid indices gracefully by skipping them
    out_files = repack_windows(Path(md.path), [[invalid_idx]], out_prefix)
    assert len(out_files) == 0  # No files created for invalid windows


def test_repack_window_contents(test_model_path, temp_cache_dir):
    """Test that repacked window contents match original weights."""
    layer_idx = 0
    md = get_model_metadata(test_model_path)

    # Repack layer into window
    out_prefix = temp_cache_dir / "test"
    out_files = repack_windows(Path(md.path), [[layer_idx]], out_prefix)
    assert len(out_files) == 1
    out_file = out_files[0]

    # Load repacked file
    repacked = mx.load(str(out_file))

    # Compare with original weights
    for k, wt in md.weight_info[layer_idx].items():
        tensor_name = f"model.layers.{layer_idx}.{k}"
        original = load_weight(wt, {})
        repacked_tensor = repacked[tensor_name]

        # Compare tensor shapes and values
        assert original.shape == repacked_tensor.shape
        assert mx.array_equal(original, repacked_tensor)
