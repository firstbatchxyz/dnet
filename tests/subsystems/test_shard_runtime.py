"""Tests: ShardRuntime init, load/unload core, KV cache, compute staging, reset."""

# ruff: noqa: E402
import time as _time
import pytest

mx = pytest.importorskip("mlx.core")

from dnet.shard.runtime import ShardRuntime
from dnet.shard.models import ShardLoadModelRequest, ShardUnloadModelResponse
from dnet.core.types.messages import ActivationMessage
from tests.fakes import FakeModelMetadata, FakeRingModel, FakePolicyPlan, FakePolicy

pytestmark = [pytest.mark.shard, pytest.mark.runtime]


def _create_request(kv_bits="8bit", layers=None):
    return ShardLoadModelRequest(
        model_path="m",
        total_layers=4,
        layers=layers or [0, 3],
        warmup=False,
        next_node=None,
        window_size=2,
        residency_size=3,
        kv_bits=kv_bits,
        api_callback_address="cb",
    )


def test_init_defaults_and_wire_dtype():
    rt = ShardRuntime("S1", queue_size=16)
    assert rt.max_queue_size == 16
    assert rt._wire_dtype_str == "float16"
    assert str(rt._wire_mx_dtype) == str(mx.float16)
    from dnet.shard.policies import NoopPolicy

    assert isinstance(rt.policy, NoopPolicy)
    assert rt.activation_recv_queue.maxsize == 16
    assert rt.activation_send_queue.maxsize == 16


def test_load_model_core_sets_policy_kv_config_and_pools(monkeypatch):
    rt = ShardRuntime("S1")
    monkeypatch.setattr(
        "dnet.shard.runtime.get_model_metadata",
        lambda _: FakeModelMetadata(
            model_type="llama", model_config={"hidden_size": 8}, num_layers=4
        ),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.get_ring_model",
        lambda *a, **k: FakeRingModel(quant_applies=True, tie_word_embeddings=False),
        raising=True,
    )

    made_caches = []

    def _make_cache(model, kv_mode, kv_bits, kv_group):
        made_caches.append((kv_mode, kv_bits, kv_group))
        return {"cache": True}

    monkeypatch.setattr("dnet.shard.runtime.make_cache", _make_cache, raising=True)

    # patch the model util calls
    calls = {"emb": 0, "norm": 0, "head": 0}
    monkeypatch.setattr(
        "dnet.shard.runtime.load_embeddings",
        lambda *a, **k: calls.__setitem__("emb", calls["emb"] + 1) or 1,
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.load_final_norm",
        lambda *a, **k: calls.__setitem__("norm", calls["norm"] + 1) or 1,
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.load_lm_head",
        lambda *a, **k: calls.__setitem__("head", calls["head"] + 1) or 1,
        raising=True,
    )

    monkeypatch.setattr(
        "dnet.shard.runtime.plan_policy",
        lambda **k: FakePolicyPlan("fit", 2, 3, False),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.make_policy",
        lambda mode, runtime, resident_windows: FakePolicy(runtime, resident_windows),
        raising=True,
    )

    rt.load_model_core(_create_request(kv_bits="8bit", layers=[0, 3]))
    assert rt.kv_cache_config.mode == "8bit" and rt.kv_cache_config.bits == 8
    assert rt.input_pool is not None and rt.output_pool is not None
    assert hasattr(rt.policy, "configured") and rt.policy.window_size == 2
    assert made_caches and made_caches[0][0] in ("8bit", "fp16", "quant")
    assert calls["emb"] == 1 and calls["norm"] == 1 and calls["head"] == 1


@pytest.mark.parametrize("bits", ["4bit", "8bit", "fp16"])
def test_load_model_core_kv_modes(monkeypatch, bits):
    rt = ShardRuntime("S2")
    monkeypatch.setattr(
        "dnet.shard.runtime.get_model_metadata",
        lambda _: FakeModelMetadata(
            model_type="llama", model_config={"hidden_size": 8}, num_layers=2
        ),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.get_ring_model",
        lambda *a, **k: FakeRingModel(quant_applies=False),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.plan_policy",
        lambda **k: FakePolicyPlan("fit", 1, 1, False),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.make_policy",
        lambda *a, **k: FakePolicy(*a[1:], **{}),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.make_cache", lambda *a, **k: {}, raising=True
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.load_embeddings", lambda *a, **k: 0, raising=True
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.load_final_norm", lambda *a, **k: 0, raising=True
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.load_lm_head", lambda *a, **k: 0, raising=True
    )
    rt.load_model_core(_create_request(kv_bits=bits, layers=[1]))
    assert rt.kv_cache_config.mode in ("4bit", "8bit", "fp16")
    assert rt.kv_cache_config.bits >= 1


def test_unload_model_core_no_model_ok():
    rt = ShardRuntime("S3")
    resp = rt.unload_model_core()
    assert isinstance(resp, ShardUnloadModelResponse) and resp.success is True


def test_unload_model_core_clears_state_and_policy(monkeypatch):
    rt = ShardRuntime("S3")
    monkeypatch.setattr(  # load minimal model
        "dnet.shard.runtime.get_model_metadata",
        lambda _: FakeModelMetadata(model_type="llama", model_config={}, num_layers=2),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.get_ring_model",
        lambda *a, **k: FakeRingModel(quant_applies=False),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.plan_policy",
        lambda **k: FakePolicyPlan("fit", 1, 1, False),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.make_policy",
        lambda *a, **k: FakePolicy(*a[1:]),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.make_cache", lambda *a, **k: {}, raising=True
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.load_embeddings", lambda *a, **k: 0, raising=True
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.load_final_norm", lambda *a, **k: 0, raising=True
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.load_lm_head", lambda *a, **k: 0, raising=True
    )
    rt.load_model_core(_create_request(layers=[0]))
    assert isinstance(rt.policy, FakePolicy) and rt.policy.window_size == 1
    try:  # enqueue a recv item, then unload should drain and clear
        rt.activation_recv_queue.put_nowait(
            ActivationMessage(
                nonce="n",
                pool_id=0,
                batch_size=1,
                shape=(1,),
                dtype="float32",
                layer_id=0,
                timestamp=0,
                node_origin="S",
                callback_url="",
            )
        )
    except Exception:
        pass
    resp = rt.unload_model_core()
    assert resp.success is True
    assert rt.model is None and rt.cache is None and rt.model_metadata is None
    assert rt.assigned_layers == [] and rt.input_pool is None and rt.output_pool is None
    from dnet.shard.policies import NoopPolicy

    assert isinstance(rt.policy, NoopPolicy)


def test_reset_cache_no_model_noop():
    rt = ShardRuntime("S4")
    rt.reset_cache()


def test_reset_cache_recreates_cache(monkeypatch):
    rt = ShardRuntime("S4")
    rt.model = object()
    count = {"n": 0}
    monkeypatch.setattr(
        "dnet.shard.runtime.make_cache",
        lambda *a, **k: count.__setitem__("n", count["n"] + 1) or {},
        raising=True,
    )
    rt.reset_cache()
    assert count["n"] == 1


def test_kv_requires_model_before_creation():
    rt = ShardRuntime("S5")
    with pytest.raises(RuntimeError):
        rt.get_or_make_kv("nonce")


def test_kv_returns_same_cache_for_same_nonce(monkeypatch):
    rt = ShardRuntime("S5")
    rt.model = object()
    created: list[dict] = []
    monkeypatch.setattr(
        "dnet.shard.runtime.make_cache",
        lambda *a, **k: created.append({}) or {},
        raising=True,
    )
    kv1 = rt.get_or_make_kv("a")
    kv2 = rt.get_or_make_kv("a")
    assert kv1 is kv2
    assert len(created) == 1  # created once for first nonce


def test_kv_expires_stale_entries_on_access(monkeypatch):
    rt = ShardRuntime("S5")
    rt.model = object()
    monkeypatch.setattr(
        "dnet.shard.runtime.make_cache", lambda *a, **k: {}, raising=True
    )
    rt._kv_by_nonce["old"] = {}  # insert a stale entry and advance time beyond TTL
    rt._kv_last_seen["old"] = _time.perf_counter() - (rt._kv_ttl_s + 1.0)
    _ = rt.get_or_make_kv("b")
    assert "old" not in rt._kv_by_nonce and "old" not in rt._kv_last_seen


def test_compute_calls_policy_process():
    rt = ShardRuntime("S6")
    from tests.fakes import FakePolicy

    rt.policy = FakePolicy(rt, resident_windows=1)
    rt.compute(
        ActivationMessage(
            nonce="n",
            pool_id=0,
            batch_size=1,
            shape=(1,),
            dtype="float32",
            layer_id=0,
            timestamp=0,
            node_origin="S",
            callback_url="",
        )
    )
    assert getattr(rt.policy, "processed", False) is True


def test_wire_dtype_bf16_mapping(monkeypatch):
    # Set env var before creating runtime to test bf16 mapping
    monkeypatch.setenv("DNET_TRANSPORT_WIRE_DTYPE", "bf16")
    # Clear the cached settings to pick up new env var
    from dnet.config import get_settings

    get_settings.cache_clear()
    rt = ShardRuntime("Sbf")
    assert rt._wire_dtype_str == "bfloat16" and str(rt._wire_mx_dtype) == str(
        mx.bfloat16
    )
    # Restore default
    get_settings.cache_clear()


def test_invalid_kv_bits_fallback(monkeypatch):
    rt = ShardRuntime("Sbad")
    rt.kv_cache_config.bits = 7
    monkeypatch.setattr(
        "dnet.shard.runtime.get_model_metadata",
        lambda _: FakeModelMetadata(model_type="llama", model_config={}, num_layers=2),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.get_ring_model",
        lambda *a, **k: FakeRingModel(quant_applies=False),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.plan_policy",
        lambda **k: FakePolicyPlan("fit", 1, 1, False),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.make_policy",
        lambda *a, **k: FakePolicy(*a[1:], **{}),
        raising=True,
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.make_cache", lambda *a, **k: {}, raising=True
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.load_embeddings", lambda *a, **k: 0, raising=True
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.load_final_norm", lambda *a, **k: 0, raising=True
    )
    monkeypatch.setattr(
        "dnet.shard.runtime.load_lm_head", lambda *a, **k: 0, raising=True
    )
    # build a minimal request-like object with an invalid kv_bits value
    req = type(
        "R",
        (),
        {
            "model_path": "m",
            "total_layers": 2,
            "layers": [1],
            "warmup": False,
            "next_node": None,
            "window_size": 1,
            "residency_size": 1,
            "kv_bits": "invalid",
            "api_callback_address": "cb",
        },
    )()
    rt.load_model_core(req)
    assert rt.kv_cache_config.mode == "fp16" and rt.kv_cache_config.bits == 7
