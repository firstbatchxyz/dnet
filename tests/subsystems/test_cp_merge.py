"""Tests for context parallelism merge attention operator."""

from __future__ import annotations

import pytest
import mlx.core as mx

from dnet.core.cp.merge_attention import (
    PartialAttentionOutput,
    merge_partial_attention,
    merge_two_partials,
)


def make_partial(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    max_score_val: float = 0.0,
    lse_val: float = 1.0,
) -> PartialAttentionOutput:
    """Helper to create a partial attention output for testing."""
    return PartialAttentionOutput(
        output=mx.random.normal((seq_len, num_heads, head_dim)),
        max_score=mx.full((seq_len, num_heads), max_score_val),
        log_sum_exp=mx.full((seq_len, num_heads), lse_val),
    )


class TestMergeTwoPartials:
    """Tests for merging two partial attention outputs."""

    def test_equal_weights(self):
        """Two partials with equal stats should produce average."""
        seq_len, num_heads, head_dim = 4, 8, 64

        # Create two partials with same max_score and lse
        p1 = PartialAttentionOutput(
            output=mx.ones((seq_len, num_heads, head_dim)),
            max_score=mx.zeros((seq_len, num_heads)),
            log_sum_exp=mx.ones((seq_len, num_heads)),
        )
        p2 = PartialAttentionOutput(
            output=mx.ones((seq_len, num_heads, head_dim)) * 3,
            max_score=mx.zeros((seq_len, num_heads)),
            log_sum_exp=mx.ones((seq_len, num_heads)),
        )

        merged = merge_two_partials(p1, p2)

        # With equal weights, should be average: (1 + 3) / 2 = 2
        expected = mx.ones((seq_len, num_heads, head_dim)) * 2
        assert mx.allclose(merged.output, expected, atol=1e-5)

    def test_different_max_scores(self):
        """Partial with higher max_score should dominate."""
        seq_len, num_heads, head_dim = 4, 8, 64

        # p1 has much higher max_score -> should dominate
        p1 = PartialAttentionOutput(
            output=mx.ones((seq_len, num_heads, head_dim)),
            max_score=mx.full((seq_len, num_heads), 10.0),
            log_sum_exp=mx.ones((seq_len, num_heads)),
        )
        p2 = PartialAttentionOutput(
            output=mx.ones((seq_len, num_heads, head_dim)) * 100,
            max_score=mx.zeros((seq_len, num_heads)),
            log_sum_exp=mx.ones((seq_len, num_heads)),
        )

        merged = merge_two_partials(p1, p2)

        # p1 should dominate (scale factor for p2 is exp(-10) ≈ 0)
        assert mx.allclose(merged.output, p1.output, atol=1e-4)

    def test_numerical_stability(self):
        """Should handle large max_score values without overflow."""
        seq_len, num_heads, head_dim = 4, 8, 64

        # Very large max scores (would overflow without proper handling)
        p1 = PartialAttentionOutput(
            output=mx.ones((seq_len, num_heads, head_dim)),
            max_score=mx.full((seq_len, num_heads), 1000.0),
            log_sum_exp=mx.ones((seq_len, num_heads)),
        )
        p2 = PartialAttentionOutput(
            output=mx.ones((seq_len, num_heads, head_dim)) * 2,
            max_score=mx.full((seq_len, num_heads), 999.0),
            log_sum_exp=mx.ones((seq_len, num_heads)),
        )

        merged = merge_two_partials(p1, p2)

        # Should not have NaN or Inf
        assert not mx.any(mx.isnan(merged.output))
        assert not mx.any(mx.isinf(merged.output))

    def test_merge_updates_stats(self):
        """Merged output should have updated max_score and lse."""
        seq_len, num_heads, head_dim = 4, 8, 64

        p1 = make_partial(seq_len, num_heads, head_dim, max_score_val=5.0, lse_val=2.0)
        p2 = make_partial(seq_len, num_heads, head_dim, max_score_val=3.0, lse_val=3.0)

        merged = merge_two_partials(p1, p2)

        # New max should be max of individual maxes
        assert mx.allclose(merged.max_score, mx.full((seq_len, num_heads), 5.0))

        # New lse should be greater than individual (log of sum of exps)
        assert mx.all(merged.log_sum_exp > p1.log_sum_exp)


class TestMergePartialAttention:
    """Tests for merging multiple partial outputs."""

    def test_empty_list_raises(self):
        """Should raise on empty list."""
        with pytest.raises(ValueError, match="Cannot merge empty"):
            merge_partial_attention([])

    def test_single_partial(self):
        """Single partial should return its output unchanged."""
        p1 = make_partial(4, 8, 64)
        result = merge_partial_attention([p1])

        assert mx.allclose(result, p1.output)

    def test_multiple_partials(self):
        """Should correctly merge multiple partials."""
        seq_len, num_heads, head_dim = 4, 8, 64

        # Create 4 partials with equal weights
        partials = []
        for i in range(4):
            p = PartialAttentionOutput(
                output=mx.full((seq_len, num_heads, head_dim), float(i + 1)),
                max_score=mx.zeros((seq_len, num_heads)),
                log_sum_exp=mx.ones((seq_len, num_heads)),
            )
            partials.append(p)

        result = merge_partial_attention(partials)

        # With equal weights: (1 + 2 + 3 + 4) / 4 = 2.5
        expected = mx.full((seq_len, num_heads, head_dim), 2.5)
        assert mx.allclose(result, expected, atol=1e-4)

    def test_associativity(self):
        """Merge should be associative: merge([a,b,c]) == merge([merge([a,b]),c])."""
        partials = [make_partial(4, 8, 64) for _ in range(4)]

        # Merge all at once
        result1 = merge_partial_attention(partials)

        # Merge pairwise
        p12 = merge_two_partials(partials[0], partials[1])
        p34 = merge_two_partials(partials[2], partials[3])
        p1234 = merge_two_partials(p12, p34)

        assert mx.allclose(result1, p1234.output, atol=1e-4)


class TestRingReductionSimulation:
    """Simulate ring reduction to verify merge correctness."""

    def test_ring_reduction_4_ranks(self):
        """Simulate 4-rank ring reduction and verify final merge."""
        seq_len, num_heads, head_dim = 8, 4, 32
        num_ranks = 4

        # Create "ground truth" partials (what each rank computes)
        rank_partials = [
            make_partial(seq_len, num_heads, head_dim) for _ in range(num_ranks)
        ]

        # Simulate ring reduction: each rank progressively merges
        # At the end, all ranks should have same result
        def ring_reduce(rank_id: int) -> mx.array:
            running = rank_partials[rank_id]
            for step in range(1, num_ranks):
                # In real ring: receive from (rank_id - step) mod N
                prev_rank = (rank_id - step) % num_ranks
                running = merge_two_partials(running, rank_partials[prev_rank])
            return running.output

        # All ranks should produce same final output
        results = [ring_reduce(r) for r in range(num_ranks)]

        for i in range(1, num_ranks):
            assert mx.allclose(results[0], results[i], atol=1e-4)

        # Should also match direct merge of all
        direct = merge_partial_attention(rank_partials)
        assert mx.allclose(results[0], direct, atol=1e-4)
