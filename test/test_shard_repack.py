"""Integration tests for shard model loading with repacking."""

import asyncio
from pathlib import Path
import tempfile

import pytest
import pytest_asyncio

from dnet.ring.shard.node import RingShardNode
from dnet.ring.shard.models import ShardLoadModelRequest


@pytest_asyncio.fixture
async def shard_node():
    """Create a test shard node."""
    node = RingShardNode(
        node_id=0,
        grpc_port=58080,
        http_port=8080,
    )
    try:
        yield node
    finally:
        # Cleanup
        await node.unload_model()


@pytest.fixture
def temp_cache_root():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        # Create the layers directory
        (path / "shard_0/layers").mkdir(parents=True, exist_ok=True)
        yield path


@pytest.mark.asyncio
async def test_shard_model_loading_with_repack(
    shard_node, temp_cache_root, monkeypatch
):
    """Test that shard correctly repacks layers during model loading."""
    # Mock cache directory to use temp dir
    monkeypatch.setattr(
        shard_node.__class__,
        "_get_cache_dir",
        lambda self: temp_cache_root / f"shard_{self.node_id}/layers",
    )

    model_path = "Qwen/Qwen3-4B-MLX-4bit"  # Small test model
    test_layers = [0, 1]  # Test with first two layers
    window_size = 2

    # Create load request
    request = ShardLoadModelRequest(
        model_path=model_path,
        total_layers=32,  # Example total layers
        layers=test_layers,
        window_size=window_size,
        api_callback_address="localhost:50051",  # Dummy address
        warmup=True,  # Request warmup during loading
    )

    # Load model
    response = await shard_node.load_model(request)

    # Verify response
    assert response.success, response.message
    assert response.layers_loaded == test_layers
    assert response.load_time_ms > 0
    assert shard_node.window_size == window_size
    assert shard_node._assigned_sorted == sorted(test_layers)
    assert shard_node.total_layers == 32

    # Verify success message indicates successful repacking
    assert "model loaded successfully" in response.message.lower()

    # Warmup should complete successfully
    assert shard_node._warmup_completed


@pytest.mark.asyncio
async def test_shard_reuses_cached_layers(shard_node, temp_cache_root, monkeypatch):
    """Test that shard reuses cached repacked layers on subsequent loads."""
    # Mock cache directory to use temp dir
    monkeypatch.setattr(
        shard_node.__class__,
        "_get_cache_dir",
        lambda self: temp_cache_root / f"shard_{self.node_id}/layers",
    )

    model_path = "Qwen/Qwen3-4B-MLX-4bit"
    test_layers = [0, 1]
    window_size = 2

    request = ShardLoadModelRequest(
        model_path=model_path,
        total_layers=32,
        layers=test_layers,
        window_size=window_size,
        api_callback_address="localhost:50051",
        warmup=True,  # Request warmup during loading
    )

    # First load - should repack and warmup
    start_time = asyncio.get_event_loop().time()
    response1 = await shard_node.load_model(request)
    first_load_time = asyncio.get_event_loop().time() - start_time
    assert response1.success, response1.message

    # Verify warmup ran
    assert shard_node._warmup_completed
    assert shard_node.window_size == window_size

    # Unload model
    await shard_node.unload_model()
    assert not shard_node._warmup_completed  # Reset on unload

    # Second load - should use cache and rewarmup
    start_time = asyncio.get_event_loop().time()
    response2 = await shard_node.load_model(request)
    second_load_time = asyncio.get_event_loop().time() - start_time

    assert response2.success, response2.message
    assert shard_node._warmup_completed  # Should rewarm

    # Should be faster since files are cached
    assert second_load_time < first_load_time


@pytest.mark.asyncio
async def test_shard_handles_repack_failure(shard_node, temp_cache_root, monkeypatch):
    """Test that shard handles layer repacking failures gracefully."""
    # Mock cache directory to use temp dir
    monkeypatch.setattr(
        shard_node.__class__,
        "_get_cache_dir",
        lambda self: temp_cache_root / f"shard_{self.node_id}/layers",
    )

    # Request layer 999 which shouldn't exist
    request = ShardLoadModelRequest(
        model_path="non_existent_model",  # Use invalid model path
        total_layers=32,
        layers=[999],
        window_size=1,
        api_callback_address="localhost:50051",
    )

    # Load should fail gracefully with repository not found error
    response = await shard_node.load_model(request)
    assert not response.success
    assert "repository not found" in response.message.lower()
    assert not response.layers_loaded
    assert not shard_node._warmup_completed  # Should not run warmup on failed load


@pytest.mark.asyncio
async def test_shard_handles_window_size_change(
    shard_node, temp_cache_root, monkeypatch
):
    """Test that shard correctly handles window size changes."""
    # Mock cache directory to use temp dir
    monkeypatch.setattr(
        shard_node.__class__,
        "_get_cache_dir",
        lambda self: temp_cache_root / f"shard_{self.node_id}/layers",
    )

    model_path = "Qwen/Qwen3-4B-MLX-4bit"
    test_layers = [0, 1, 2, 3]

    # First load with window_size=2
    request1 = ShardLoadModelRequest(
        model_path=model_path,
        total_layers=32,
        layers=test_layers,
        window_size=2,  # Two layers per window
        api_callback_address="localhost:50051",
        warmup=True,  # Request warmup during loading
    )

    response1 = await shard_node.load_model(request1)
    assert response1.success, response1.message
    assert shard_node.window_size == 2

    # Verify first window size worked
    assert "model loaded successfully" in response1.message.lower()

    # Unload and reload with window_size=4
    await shard_node.unload_model()

    # Second load with larger window size
    request2 = ShardLoadModelRequest(
        model_path=model_path,
        total_layers=32,
        layers=test_layers,
        window_size=4,  # All layers in one window
        api_callback_address="localhost:50051",
        warmup=True,  # Request warmup during loading
    )

    response2 = await shard_node.load_model(request2)
    assert response2.success, response2.message
    assert shard_node.window_size == 4

    # Verify second window size worked
    assert "model loaded successfully" in response2.message.lower()
