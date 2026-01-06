"""Tests for context parallelism sharding utilities."""

from __future__ import annotations

import pytest
import mlx.core as mx

from dnet.core.cp.sharding import shard_for_mode, unshard


class TestShardForModePrefill:
    """Tests for prefill (2N load-balanced) sharding."""

    def test_basic_4_ranks(self):
        """Test 2N sharding with 4 ranks produces correct assignments."""
        # 16 tokens, 4 ranks -> 8 chunks -> pairs (0,7), (1,6), (2,5), (3,4)
        tokens = mx.arange(16)
        num_ranks = 4

        # Rank 0 gets chunks 0 and 7
        sharded, indices = shard_for_mode(tokens, num_ranks, 0, "prefill")
        assert sharded.shape[0] == 4  # 2 + 2 tokens
        assert indices == [0, 1, 14, 15]

        # Rank 1 gets chunks 1 and 6
        sharded, indices = shard_for_mode(tokens, num_ranks, 1, "prefill")
        assert indices == [2, 3, 12, 13]

        # Rank 3 gets chunks 3 and 4 (middle)
        sharded, indices = shard_for_mode(tokens, num_ranks, 3, "prefill")
        assert indices == [6, 7, 8, 9]

    def test_load_balance(self):
        """Verify all ranks get equal-sized chunks (load balanced)."""
        tokens = mx.arange(64)
        num_ranks = 4

        sizes = []
        for rank_id in range(num_ranks):
            sharded, _ = shard_for_mode(tokens, num_ranks, rank_id, "prefill")
            sizes.append(sharded.shape[0])

        # All sizes should be equal (or differ by at most 1 for remainders)
        assert max(sizes) - min(sizes) <= 1

    def test_single_rank(self):
        """Single rank should get all tokens."""
        tokens = mx.arange(10)
        sharded, indices = shard_for_mode(tokens, 1, 0, "prefill")

        assert sharded.shape[0] == 10
        assert indices == list(range(10))

    def test_coverage_all_indices(self):
        """All indices should be covered exactly once across all ranks."""
        tokens = mx.arange(32)
        num_ranks = 4

        all_indices = []
        for rank_id in range(num_ranks):
            _, indices = shard_for_mode(tokens, num_ranks, rank_id, "prefill")
            all_indices.extend(indices)

        assert sorted(all_indices) == list(range(32))


class TestShardForModeDecode:
    """Tests for decode (even N-way) sharding."""

    def test_basic_4_ranks(self):
        """Test even sharding with 4 ranks."""
        tokens = mx.arange(16)
        num_ranks = 4

        # Each rank gets contiguous 4 tokens
        for rank_id in range(num_ranks):
            sharded, indices = shard_for_mode(tokens, num_ranks, rank_id, "decode")
            assert sharded.shape[0] == 4
            assert indices == list(range(rank_id * 4, (rank_id + 1) * 4))

    def test_uneven_split(self):
        """Test handling of sequence length not divisible by ranks."""
        tokens = mx.arange(10)
        num_ranks = 4

        all_indices = []
        for rank_id in range(num_ranks):
            sharded, indices = shard_for_mode(tokens, num_ranks, rank_id, "decode")
            all_indices.extend(indices)

        # All indices covered
        assert sorted(all_indices) == list(range(10))

    def test_contiguous_chunks(self):
        """Decode sharding should produce contiguous chunks."""
        tokens = mx.arange(100)
        num_ranks = 4

        for rank_id in range(num_ranks):
            _, indices = shard_for_mode(tokens, num_ranks, rank_id, "decode")
            # Check contiguity: indices should be sequential
            for i in range(1, len(indices)):
                assert indices[i] == indices[i - 1] + 1


class TestShardValidation:
    """Tests for input validation."""

    def test_invalid_num_ranks(self):
        """Should raise on invalid num_ranks."""
        tokens = mx.arange(10)
        with pytest.raises(ValueError, match="num_ranks must be positive"):
            shard_for_mode(tokens, 0, 0, "prefill")

    def test_rank_out_of_range(self):
        """Should raise on rank_id out of range."""
        tokens = mx.arange(10)
        with pytest.raises(ValueError, match="rank_id .* out of range"):
            shard_for_mode(tokens, 4, 5, "prefill")

    def test_empty_input(self):
        """Empty input should return empty output."""
        tokens = mx.zeros((0, 128))
        sharded, indices = shard_for_mode(tokens, 4, 0, "prefill")
        assert sharded.shape[0] == 0
        assert indices == []


class TestUnshard:
    """Tests for unshard operation."""

    def test_roundtrip_prefill(self):
        """Shard -> unshard should recover original."""
        original = mx.arange(32).reshape(32, 1).astype(mx.float32)
        num_ranks = 4

        # Shard
        chunks = []
        indices_list = []
        for rank_id in range(num_ranks):
            sharded, indices = shard_for_mode(original, num_ranks, rank_id, "prefill")
            chunks.append(sharded)
            indices_list.append(indices)

        # Unshard
        recovered = unshard(chunks, indices_list, 32)

        assert mx.allclose(recovered, original)

    def test_roundtrip_decode(self):
        """Shard -> unshard should recover original for decode mode."""
        original = mx.arange(32).reshape(32, 1).astype(mx.float32)
        num_ranks = 4

        chunks = []
        indices_list = []
        for rank_id in range(num_ranks):
            sharded, indices = shard_for_mode(original, num_ranks, rank_id, "decode")
            chunks.append(sharded)
            indices_list.append(indices)

        recovered = unshard(chunks, indices_list, 32)

        assert mx.allclose(recovered, original)

    def test_multidimensional(self):
        """Test with multi-dimensional tensors."""
        # Simulate hidden states: [seq, heads, dim]
        original = mx.random.normal((64, 8, 128))
        num_ranks = 4

        chunks = []
        indices_list = []
        for rank_id in range(num_ranks):
            sharded, indices = shard_for_mode(original, num_ranks, rank_id, "decode")
            chunks.append(sharded)
            indices_list.append(indices)

        recovered = unshard(chunks, indices_list, 64)

        assert mx.allclose(recovered, original, atol=1e-5)
