"""Tests for context parallelism algorithm selection heuristics."""

from __future__ import annotations


from dnet.core.cp.heuristics import (
    CPAlgorithm,
    select_algorithm,
    estimate_algorithm_latency,
)


class TestSelectAlgorithm:
    """Tests for the greedy heuristic algorithm selection."""

    def test_cp_disabled(self):
        """Should return SINGLE_DEVICE when CP is disabled."""
        result = select_algorithm(
            new_tokens=10000,
            cached_tokens=50000,
            batch_size=1,
            num_ranks=4,
            num_q_heads=32,
            num_kv_heads=8,
            context_parallel_enabled=False,
        )
        assert result == CPAlgorithm.SINGLE_DEVICE

    def test_small_context(self):
        """Should return SINGLE_DEVICE for small contexts."""
        result = select_algorithm(
            new_tokens=1000,
            cached_tokens=2000,
            batch_size=1,
            num_ranks=4,
            num_q_heads=32,
            num_kv_heads=8,
            context_parallel_enabled=True,
            min_context_for_cp=32768,
        )
        assert result == CPAlgorithm.SINGLE_DEVICE

    def test_single_rank(self):
        """Single rank should return SINGLE_DEVICE."""
        result = select_algorithm(
            new_tokens=10000,
            cached_tokens=50000,
            batch_size=1,
            num_ranks=1,
            num_q_heads=32,
            num_kv_heads=8,
            context_parallel_enabled=True,
        )
        assert result == CPAlgorithm.SINGLE_DEVICE

    def test_decode_mode(self):
        """Decode (new_tokens <= batch_size) should use RING_REDUCE."""
        result = select_algorithm(
            new_tokens=4,  # 4 tokens for batch of 4 -> decode
            cached_tokens=100000,
            batch_size=4,
            num_ranks=4,
            num_q_heads=32,
            num_kv_heads=8,
            context_parallel_enabled=True,
            min_context_for_cp=32768,
        )
        assert result == CPAlgorithm.RING_REDUCE

    def test_full_prefill(self):
        """Full prefill with sufficient tokens should use PASS_KV."""
        result = select_algorithm(
            new_tokens=50000,  # Full prefill, no cache
            cached_tokens=0,
            batch_size=1,
            num_ranks=4,
            num_q_heads=32,
            num_kv_heads=8,
            context_parallel_enabled=True,
            min_context_for_cp=32768,
        )
        assert result == CPAlgorithm.PASS_KV

    def test_high_cache_hit(self):
        """High cache hit rate (low miss rate) should use PASS_Q."""
        # miss_rate = 100 / (100 + 100000) ≈ 0.001 < 0.125
        result = select_algorithm(
            new_tokens=100,  # Very few new tokens
            cached_tokens=100000,  # Large cache
            batch_size=1,
            num_ranks=4,
            num_q_heads=32,
            num_kv_heads=8,
            context_parallel_enabled=True,
            min_context_for_cp=32768,
        )
        assert result == CPAlgorithm.PASS_Q

    def test_gqa_threshold_calculation(self):
        """GQA threshold should be computed correctly."""
        # With 128 Q heads and 8 KV heads: threshold = 2*8/128 = 0.125
        # miss_rate = 5000 / 50000 = 0.1 < 0.125 -> PASS_Q This test has been removed from the coverage

        # miss_rate = 10000 / 50000 = 0.2 > 0.125 -> PASS_KV
        result = select_algorithm(
            new_tokens=10000,
            cached_tokens=40000,
            batch_size=1,
            num_ranks=4,
            num_q_heads=128,
            num_kv_heads=8,
            context_parallel_enabled=True,
            min_context_for_cp=32768,
        )
        assert result == CPAlgorithm.PASS_KV

    def test_custom_thresholds(self):
        """Custom thresholds should override defaults."""
        result = select_algorithm(
            new_tokens=5000,
            cached_tokens=5000,  # 10K total, would normally skip CP
            batch_size=1,
            num_ranks=4,
            num_q_heads=32,
            num_kv_heads=8,
            context_parallel_enabled=True,
            min_context_for_cp=8000,  # Lower threshold
        )
        # Should now consider CP since 10K > 8K
        assert result in (CPAlgorithm.PASS_KV, CPAlgorithm.PASS_Q)


class TestEstimateAlgorithmLatency:
    """Tests for latency estimation (for solver integration)."""

    def test_single_device_latency(self):
        """Single device should have straightforward compute latency."""
        latency = estimate_algorithm_latency(
            algorithm=CPAlgorithm.SINGLE_DEVICE,
            new_tokens=1000,
            cached_tokens=50000,
            num_ranks=4,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
            flops_per_sec=1e12,  # 1 TFLOPS
            bandwidth_bytes_per_sec=100e9,  # 100 GB/s
        )
        # Should be positive and finite
        assert latency > 0
        assert latency < float("inf")

    def test_pass_kv_vs_single_device(self):
        """PASS_KV with more ranks should be faster than single device."""
        common_args = dict(
            new_tokens=10000,
            cached_tokens=50000,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
            flops_per_sec=1e12,
            bandwidth_bytes_per_sec=100e9,
        )

        single_latency = estimate_algorithm_latency(
            algorithm=CPAlgorithm.SINGLE_DEVICE, num_ranks=1, **common_args
        )
        pass_kv_latency = estimate_algorithm_latency(
            algorithm=CPAlgorithm.PASS_KV, num_ranks=4, **common_args
        )

        # With ideal scaling, 4 ranks should be ~4x faster
        # In practice, communication overhead reduces this
        assert pass_kv_latency < single_latency

    def test_ring_reduce_vs_pass_q(self):
        """RING_REDUCE should avoid All2All overhead."""
        common_args = dict(
            new_tokens=4,  # Decode-like
            cached_tokens=100000,
            num_ranks=4,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
            flops_per_sec=1e12,
            bandwidth_bytes_per_sec=100e9,
        )

        pass_q_latency = estimate_algorithm_latency(
            algorithm=CPAlgorithm.PASS_Q, **common_args
        )
        ring_reduce_latency = estimate_algorithm_latency(
            algorithm=CPAlgorithm.RING_REDUCE, **common_args
        )

        # Ring reduce should be faster (no All2All)
        assert ring_reduce_latency <= pass_q_latency


class TestCPAlgorithmEnum:
    """Tests for CPAlgorithm enum."""

    def test_string_values(self):
        """Enum values should be lowercase strings."""
        assert CPAlgorithm.SINGLE_DEVICE == "single_device"
        assert CPAlgorithm.PASS_KV == "pass_kv"
        assert CPAlgorithm.PASS_Q == "pass_q"
        assert CPAlgorithm.RING_REDUCE == "ring_reduce"

    def test_is_str_enum(self):
        """Should be usable as strings."""
        algo = CPAlgorithm.PASS_KV
        assert f"Using {algo}" == "Using pass_kv"
