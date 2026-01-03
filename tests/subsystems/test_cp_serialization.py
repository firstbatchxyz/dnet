import sys
from unittest.mock import MagicMock

# Mock dnet.compression to avoid Metal dependency on Linux
mock_compression = MagicMock()
sys.modules["dnet.compression"] = mock_compression
sys.modules["dnet.compression.ops"] = MagicMock()
sys.modules["dnet.compression.kernels"] = MagicMock()

import pytest  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from dnet.shard.adapters.context_parallel import CPAdapter  # noqa: E402
from dnet.core.cp.merge_attention import PartialAttentionOutput  # noqa: E402


# Mock dependencies for CPAdapter init
class MockRuntime:
    max_queue_size = 10


class MockDiscovery:
    pass


@pytest.fixture
def adapter():
    return CPAdapter(runtime=MockRuntime(), discovery=MockDiscovery())


def test_kv_serialization_roundtrip(adapter):
    # Create test tensors
    k = mx.random.uniform(shape=(2, 4, 32))
    v = mx.random.uniform(shape=(2, 4, 32))

    # Serialize
    data = adapter._serialize_kv(k, v)
    assert isinstance(data, bytes)
    assert len(data) > 0

    # Deserialize
    k_out, v_out = adapter._deserialize_kv(data)

    # Verify
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape
    assert k_out.dtype == k.dtype
    assert v_out.dtype == v.dtype

    # Check values (using numpy for comparison)
    np.testing.assert_allclose(np.array(k_out), np.array(k), rtol=1e-5)
    np.testing.assert_allclose(np.array(v_out), np.array(v), rtol=1e-5)


def test_partial_serialization_roundtrip(adapter):
    # Create test partial output
    out = mx.random.uniform(shape=(2, 8, 64))
    # Max score: [B, H]
    max_s = mx.random.uniform(shape=(2, 8))
    lse = mx.random.uniform(shape=(2, 8))

    partial = PartialAttentionOutput(output=out, max_score=max_s, log_sum_exp=lse)

    # Serialize
    data = adapter._serialize_partial(partial)
    assert isinstance(data, bytes)

    # Deserialize
    p_out = adapter._deserialize_partial(data)

    # Verify output
    assert p_out.output.shape == out.shape
    np.testing.assert_allclose(np.array(p_out.output), np.array(out), rtol=1e-5)

    # Verify metadata (restored shape)
    assert p_out.max_score.shape == max_s.shape
    assert p_out.log_sum_exp.shape == lse.shape

    np.testing.assert_allclose(np.array(p_out.max_score), np.array(max_s), rtol=1e-5)
    np.testing.assert_allclose(np.array(p_out.log_sum_exp), np.array(lse), rtol=1e-5)
