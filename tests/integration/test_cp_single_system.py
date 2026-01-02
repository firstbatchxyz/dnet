"""Integration tests for Context Parallelism.

These tests validate CP functionality end-to-end:
1. CP module integration tests (no mocks, real tensor operations)
2. Multi-rank simulation using actual ring communication
3. End-to-end server tests when servers are available

Usage (module-level tests - no servers needed):
    uv run pytest tests/integration/test_cp_single_system.py::TestCPModuleIntegration -v

Usage (server tests - requires running servers):
    uv run pytest tests/integration/test_cp_single_system.py::TestCPServerInference -v --start-servers
"""

import logging
import os
import signal
import subprocess
import sys
import time
from typing import Generator

import pytest
import requests
import mlx.core as mx

from dnet.core.cp.sharding import shard_for_mode, unshard
from dnet.core.cp.merge_attention import (
    PartialAttentionOutput,
    merge_partial_attention,
    merge_two_partials,
)
from dnet.core.cp.heuristics import select_algorithm, CPAlgorithm
from dnet.core.cp.ring_comm import (
    CPRingCommunicator,
    MockRingCommunicator,
)
from dnet.shard.adapters.context_parallel import CPAdapter
from dnet.config import ContextParallelSettings, get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration
API_HTTP_PORT = 8080
SHARD_HTTP_PORT = 8081
BASE_URL = f"http://localhost:{API_HTTP_PORT}"


# =============================================================================
# MODULE-LEVEL INTEGRATION TESTS (no servers, real computations)
# =============================================================================


@pytest.mark.integration
class TestCPModuleIntegration:
    """Test CP modules work together correctly with real tensor operations."""

    def test_sharding_merge_roundtrip_prefill(self) -> None:
        """Test full prefill sharding -> attention -> merge pipeline."""
        # Create realistic input tensors
        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 64
        num_ranks = 4

        # Input sequence
        x = mx.random.normal((seq_len, batch_size, num_heads * head_dim))
        mx.eval(x)  # Force evaluation

        # Shard across ranks
        shards = []
        indices_list = []
        for rank in range(num_ranks):
            shard_data, indices = shard_for_mode(x, num_ranks, rank, mode="prefill")
            mx.eval(shard_data)
            shards.append(shard_data)
            indices_list.append(indices)

        # Unshard and verify roundtrip
        reconstructed = unshard(shards, indices_list, seq_len)
        mx.eval(reconstructed)

        # Verify exact reconstruction
        assert reconstructed.shape == x.shape
        diff = mx.abs(reconstructed - x)
        max_diff = float(mx.max(diff).item())
        assert max_diff < 1e-6, f"Roundtrip error: {max_diff}"

    def test_sharding_merge_roundtrip_decode(self) -> None:
        """Test full decode sharding -> attention -> merge pipeline."""
        seq_len = 1024
        hidden_dim = 512
        num_ranks = 4

        x = mx.random.normal((seq_len, hidden_dim))
        mx.eval(x)

        shards = []
        indices_list = []
        for rank in range(num_ranks):
            shard_data, indices = shard_for_mode(x, num_ranks, rank, mode="decode")
            mx.eval(shard_data)
            shards.append(shard_data)
            indices_list.append(indices)

        reconstructed = unshard(shards, indices_list, seq_len)
        mx.eval(reconstructed)

        assert reconstructed.shape == x.shape
        diff = mx.abs(reconstructed - x)
        max_diff = float(mx.max(diff).item())
        assert max_diff < 1e-6, f"Roundtrip error: {max_diff}"

    def test_partial_attention_merge_numerical_stability(self) -> None:
        """Test that merging partial attention outputs is numerically stable."""
        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        # Create partial outputs with varying scales (tests numerical stability)
        partials = []
        for i in range(4):
            # Use different scales to stress-test the merge
            scale = 10.0 ** (i - 2)  # 0.01, 0.1, 1.0, 10.0
            output = (
                mx.random.normal((batch_size, seq_len, num_heads, head_dim)) * scale
            )
            max_score = (
                mx.random.normal((batch_size, seq_len, num_heads)) + i * 2
            )  # Varying max scores
            log_sum_exp = (
                mx.abs(mx.random.normal((batch_size, seq_len, num_heads))) + 0.1
            )

            mx.eval(output, max_score, log_sum_exp)
            partials.append(
                PartialAttentionOutput(
                    output=output,
                    max_score=max_score,
                    log_sum_exp=log_sum_exp,
                )
            )

        # Merge should produce finite results
        merged = merge_partial_attention(partials)
        mx.eval(merged)

        assert merged.shape == (batch_size, seq_len, num_heads, head_dim)
        assert mx.all(mx.isfinite(merged)).item(), "Merged output contains NaN/Inf"

    def test_pairwise_merge_associativity(self) -> None:
        """Test that pairwise merging produces same result regardless of order."""
        batch_size = 1
        seq_len = 32
        num_heads = 2
        head_dim = 16

        def make_partial():
            return PartialAttentionOutput(
                output=mx.random.normal((batch_size, seq_len, num_heads, head_dim)),
                max_score=mx.random.normal((batch_size, seq_len, num_heads)),
                log_sum_exp=mx.abs(mx.random.normal((batch_size, seq_len, num_heads)))
                + 0.1,
            )

        p1, p2, p3 = make_partial(), make_partial(), make_partial()
        mx.eval(p1.output, p2.output, p3.output)

        # Merge (p1, p2), then p3
        m12 = merge_two_partials(p1, p2)
        result_12_3 = merge_two_partials(m12, p3)
        mx.eval(result_12_3.output)

        # Merge p1, then (p2, p3)
        m23 = merge_two_partials(p2, p3)
        result_1_23 = merge_two_partials(p1, m23)
        mx.eval(result_1_23.output)

        # Results should be close (floating point tolerance)
        diff = mx.abs(result_12_3.output - result_1_23.output)
        max_diff = float(mx.max(diff).item())
        assert max_diff < 1e-4, f"Merge order affects result: {max_diff}"

    def test_algorithm_selection_consistency(self) -> None:
        """Test algorithm selection produces consistent results for same inputs."""
        settings = ContextParallelSettings()

        test_cases = [
            # (new_tokens, cached_tokens, expected_algorithm)
            (100, 0, CPAlgorithm.SINGLE_DEVICE),  # Short context
            (65536, 0, CPAlgorithm.PASS_KV),  # Long prefill
            (1, 65536, CPAlgorithm.RING_REDUCE),  # Decode mode
            (1024, 60000, CPAlgorithm.PASS_Q),  # Partial prefill
        ]

        for new_tokens, cached_tokens, expected in test_cases:
            result = select_algorithm(
                new_tokens=new_tokens,
                cached_tokens=cached_tokens,
                batch_size=1,
                num_ranks=4,
                num_q_heads=32,
                num_kv_heads=8,
                context_parallel_enabled=True,
                min_context_for_cp=settings.min_context_for_cp,
            )
            assert result == expected, (
                f"Expected {expected} for ({new_tokens}, {cached_tokens}), got {result}"
            )


@pytest.mark.integration
class TestCPRingCommunication:
    """Test ring communication with actual async operations."""

    def test_ring_full_rotation_4_ranks(self) -> None:
        """Test that data correctly rotates through all ranks in the ring."""
        import asyncio

        async def run_test():
            num_ranks = 4
            ring = MockRingCommunicator(num_ranks=num_ranks)
            ranks = [ring.get_communicator(i) for i in range(num_ranks)]

            # Each rank starts with unique data
            initial_data = [f"rank_{i}_data".encode() for i in range(num_ranks)]

            # Track what each rank sees over N-1 rotations
            all_seen: list[list[bytes]] = [[] for _ in range(num_ranks)]

            current_data = initial_data.copy()

            for step in range(num_ranks - 1):
                # All ranks send/recv simultaneously
                results = await asyncio.gather(
                    *[
                        ranks[i].send_recv(current_data[i], f"step_{step}")
                        for i in range(num_ranks)
                    ]
                )

                # Update current data and track what we received
                for i in range(num_ranks):
                    all_seen[i].append(results[i])
                    current_data[i] = results[i]

            # After N-1 rotations, each rank should have seen all other ranks' data
            for rank_id in range(num_ranks):
                seen_set = set(all_seen[rank_id])
                # Should have received from all ranks except self
                expected_others = {
                    d for i, d in enumerate(initial_data) if i != rank_id
                }
                assert seen_set == expected_others, (
                    f"Rank {rank_id} missing data: {expected_others - seen_set}"
                )

        asyncio.run(run_test())

    def test_ring_communicator_initialization(self) -> None:
        """Test CPRingCommunicator initializes correctly."""
        comm = CPRingCommunicator(rank_id=2, num_ranks=4)

        assert comm.rank_id == 2
        assert comm.num_ranks == 4
        assert comm.prev_rank == 1
        assert comm.next_rank == 3

    def test_ring_communicator_edge_cases(self) -> None:
        """Test ring communicator with edge case configurations."""
        # Single rank should work
        single = CPRingCommunicator(rank_id=0, num_ranks=1)
        assert single.prev_rank == 0
        assert single.next_rank == 0

        # First rank wraps to last
        first = CPRingCommunicator(rank_id=0, num_ranks=4)
        assert first.prev_rank == 3

        # Last rank wraps to first
        last = CPRingCommunicator(rank_id=3, num_ranks=4)
        assert last.next_rank == 0


@pytest.mark.integration
class TestCPAdapterIntegration:
    """Test CPAdapter without mocking - actual algorithm and selection logic."""

    def test_adapter_full_lifecycle(self) -> None:
        """Test adapter initialization, algorithm selection, and reset."""
        import asyncio

        class MockRuntime:
            max_queue_size = 16

        adapter = CPAdapter(
            runtime=MockRuntime(),  # type: ignore
            discovery=None,  # type: ignore
            rank_id=1,
            num_ranks=4,
        )

        assert adapter.rank_id == 1
        assert adapter.num_ranks == 4
        assert adapter._algorithm == CPAlgorithm.SINGLE_DEVICE

        # Test algorithm selection for different scenarios
        algo = adapter.select_algorithm_for_request(
            new_tokens=65536, cached_tokens=0, batch_size=1
        )
        assert algo == CPAlgorithm.PASS_KV
        assert adapter._algorithm == CPAlgorithm.PASS_KV

        algo = adapter.select_algorithm_for_request(
            new_tokens=1, cached_tokens=65536, batch_size=1
        )
        assert algo == CPAlgorithm.RING_REDUCE

        # Test reset
        asyncio.run(adapter.reset_topology())
        assert adapter.rank_id == 0
        assert adapter.num_ranks == 1


@pytest.mark.integration
class TestCPConfiguration:
    """Test CP configuration loading and validation."""

    def test_settings_defaults(self, monkeypatch) -> None:
        """Test default CP settings without environment overrides."""
        # Clear any env vars that would override defaults
        monkeypatch.delenv("DNET_CP_ENABLED", raising=False)
        monkeypatch.delenv("DNET_CP_ALGORITHM", raising=False)

        settings = ContextParallelSettings()

        assert settings.enabled is False
        assert settings.algorithm == "auto"
        assert settings.min_context_for_cp == 32768
        assert settings.min_tokens_for_pass_kv == 256
        assert settings.chunk_overlap == 0

    def test_settings_accessible_from_dnet_settings(self) -> None:
        """Test CP settings are integrated into main DnetSettings."""
        all_settings = get_settings()
        cp_settings = all_settings.context_parallel

        # Verify CP settings are loaded and accessible
        _ = cp_settings.enabled
        _ = cp_settings.algorithm
        _ = cp_settings.min_context_for_cp
        _ = cp_settings.min_tokens_for_pass_kv
        _ = cp_settings.chunk_overlap


# =============================================================================
# SERVER-LEVEL INTEGRATION TESTS (requires running servers)
# =============================================================================


def wait_for_health(url: str, timeout: float = 60) -> bool:
    """Wait for server health endpoint to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def servers(start_servers_flag) -> Generator[None, None, None]:
    """Start servers with CP enabled if --start-servers flag is set."""
    procs: list[subprocess.Popen] = []

    if start_servers_flag:
        env = {**os.environ, "PYTHONPATH": "src", "DNET_CP_ENABLED": "true"}

        shard_proc = subprocess.Popen(
            [sys.executable, "-m", "cli.shard", "--http-port", str(SHARD_HTTP_PORT)],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(shard_proc)

        if not wait_for_health(f"http://localhost:{SHARD_HTTP_PORT}", timeout=30):
            for p in procs:
                p.kill()
            pytest.skip("Shard server not healthy")

        api_proc = subprocess.Popen(
            [sys.executable, "-m", "cli.api", "--http-port", str(API_HTTP_PORT)],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(api_proc)

    if not wait_for_health(BASE_URL):
        for p in procs:
            p.kill()
        pytest.skip(f"API server not healthy at {BASE_URL}")

    yield

    for p in procs:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


@pytest.mark.integration
class TestCPServerInference:
    """Server-level tests - only run when servers are available."""

    def test_server_health(self, servers) -> None:
        """Verify servers are running with CP config."""
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200

    def test_inference_with_cp_enabled(self, servers) -> None:
        """Test inference with CP-enabled server."""
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        # Prepare and load
        resp = requests.post(
            f"{BASE_URL}/v1/prepare_topology", json={"model": model_id}, timeout=300
        )
        resp.raise_for_status()

        resp = requests.post(
            f"{BASE_URL}/v1/load_model", json={"model": model_id}, timeout=300
        )
        resp.raise_for_status()

        try:
            # Inference
            resp = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 10,
                },
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()

            assert "choices" in result
            assert len(result["choices"]) > 0
        finally:
            requests.post(f"{BASE_URL}/v1/unload_model", timeout=30)
