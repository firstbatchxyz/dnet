"""Tests: InferenceManager ring connect and generate_stream flow/validation."""

import asyncio
import pytest
from pydantic import ValidationError

pytest.importorskip("mlx.core")

from dnet.api.inference import InferenceManager  # noqa: E402
from dnet.api.models import ChatRequestModel, ChatMessage  # noqa: E402
from tests.fakes import (  # noqa: E402
    FakeTokenizer,
    FakeTokenizerWithTemplate,
    FakeTokenResult,
    FakeStrategyAdapter,
    FakeClusterManager,
    FakeModelManager,
)

# Mark this module as API
pytestmark = [pytest.mark.api]


def test_connect_sets_callback_addr():
    async def main():
        cm = FakeClusterManager()
        mm = FakeModelManager(FakeTokenizer())
        ad = FakeStrategyAdapter()
        mgr = InferenceManager(cm, mm, grpc_port=50500, adapter=ad)
        await mgr.connect_to_ring("1.2.3.4", 9000, "9.9.9.9:50500")
        assert ad.connected == ("1.2.3.4", 9000)
        assert mgr._api_callback_addr == "9.9.9.9:50500"

    asyncio.run(main())


def test_generate_stream_basic_flow():
    async def main():
        tok = FakeTokenizer()
        mm = FakeModelManager(tok)
        ad = FakeStrategyAdapter()
        cm = FakeClusterManager()
        mgr = InferenceManager(cm, mm, grpc_port=50500, adapter=ad)
        await mgr.connect_to_ring("1.2.3.4", 9000, "9.9.9.9:50500")
        req = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=5,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5,
            min_p=0.2,
            min_tokens_to_keep=3,
            logprobs=True,
            top_logprobs=2,
        )
        agen = mgr.generate_stream(req)
        c0 = await agen.__anext__()
        assert c0.choices[0].delta.role == "assistant"
        assert ad.reset is True
        nonce = c0.id
        ad.queue_token(nonce, FakeTokenResult(1))
        ad.queue_token(nonce, FakeTokenResult(tok.eos_token_id))
        c1 = await agen.__anext__()
        assert c1.choices[0].delta.content == "t1"
        assert len(ad.sent) >= 1
        first_send = ad.sent[0]
        assert first_send["callback_addr"] == "9.9.9.9:50500"
        dc = first_send["decoding_config"]
        assert dc.temperature == 0.7
        assert dc.top_p == 0.9
        assert dc.repetition_penalty == 1.5
        assert dc.min_p == 0.2
        assert dc.min_tokens_to_keep == 3
        assert first_send["logprobs"] is True
        assert first_send["top_logprobs"] == 2
        expected_prompt = "hi\nAssistant:"
        assert len(first_send["tokens"]) == len(expected_prompt) * 4
        c2 = await agen.__anext__()
        assert c2.choices[0].finish_reason is None
        c3 = await agen.__anext__()
        assert c3.choices[0].finish_reason.value == "stop"

    asyncio.run(main())


def test_generate_stream_requires_tokenizer():
    async def main():
        cm = FakeClusterManager()
        mm = FakeModelManager(None)
        ad = FakeStrategyAdapter()
        mgr = InferenceManager(cm, mm, grpc_port=0, adapter=ad)
        req = ChatRequestModel(
            model="m", messages=[ChatMessage(role="user", content="hi")], logprobs=True
        )
        with pytest.raises(RuntimeError):
            _ = await mgr.generate_stream(req).__anext__()

    asyncio.run(main())


def test_chat_completions_aggregates():
    async def main():
        tok = FakeTokenizer()
        mm = FakeModelManager(tok)
        ad = FakeStrategyAdapter()
        cm = FakeClusterManager()
        mgr = InferenceManager(cm, mm, grpc_port=0, adapter=ad)
        req = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            max_tokens=3,
            logprobs=True,
            top_logprobs=1,
            profile=True,
        )
        ad.queue_token("any", FakeTokenResult(7, logprob=-0.1, top_logprobs={7: -0.1}))
        ad.queue_token(
            "any",
            FakeTokenResult(
                tok.eos_token_id, logprob=-0.2, top_logprobs={tok.eos_token_id: -0.2}
            ),
        )
        resp = await mgr.chat_completions(req)
        choice = resp.choices[0]
        assert choice.message.content == "t7t99"
        assert choice.logprobs is not None
        assert choice.logprobs.token_logprobs == [-0.1, -0.2]
        assert choice.logprobs.tokens == [7, tok.eos_token_id]
        assert isinstance(resp.metrics, dict)
        for k in (
            "total_ms",
            "ttfb_ms",
            "token_gen_ms",
            "tokens_generated",
            "tps_overall",
            "tps_decoding",
        ):
            assert k in resp.metrics
        assert resp.usage is not None
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    asyncio.run(main())


def test_resolve_request():
    tok = FakeTokenizer()
    mm = FakeModelManager(tok)
    ad = FakeStrategyAdapter()
    cm = FakeClusterManager()
    mgr = InferenceManager(cm, mm, grpc_port=0, adapter=ad)
    mgr.resolve_request("abc", {"ok": True})
    assert ad.resolved["abc"]["ok"] is True


def test_generate_stream_uses_chat_template_when_present():
    async def main():
        tok = FakeTokenizerWithTemplate()
        mm = FakeModelManager(tok)
        ad = FakeStrategyAdapter()
        cm = FakeClusterManager()
        mgr = InferenceManager(cm, mm, grpc_port=4040, adapter=ad)
        await mgr.connect_to_ring("1.2.3.4", 9000, "9.9.9.9:4040")
        req = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            max_tokens=1,
            logprobs=True,
        )
        agen = mgr.generate_stream(req)
        c0 = await agen.__anext__()
        nonce = c0.id
        ad.queue_token(nonce, FakeTokenResult(1))
        _ = await agen.__anext__()
        assert len(ad.sent) == 1
        first_send = ad.sent[0]
        assert (len(first_send["tokens"]) // 4) == len("<applied-chat-template>")
        assert tok.applied_with is not None
        assert tok.applied_with["add_generation_prompt"] is True
        assert tok.applied_with["tokenize"] is False
        assert tok.applied_with["message_dicts"][0] == ("user", "x")

    asyncio.run(main())


def test_generate_stream_length_finish_when_reaching_max_tokens():
    async def main():
        tok = FakeTokenizer()
        mm = FakeModelManager(tok)
        ad = FakeStrategyAdapter()
        cm = FakeClusterManager()
        mgr = InferenceManager(cm, mm, grpc_port=0, adapter=ad)
        req = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=3,
            logprobs=True,
        )
        agen = mgr.generate_stream(req)
        c0 = await agen.__anext__()
        nonce = c0.id
        for t in (1, 2, 3):
            ad.queue_token(nonce, FakeTokenResult(t))
            _ = await agen.__anext__()
        c_last = await agen.__anext__()
        assert c_last.choices[0].finish_reason.value == "length"

    asyncio.run(main())


def test_invalid_request_params_temperature_range():
    with pytest.raises(ValidationError):
        _ = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            temperature=-0.1,
            logprobs=True,
        )
    with pytest.raises(ValidationError):
        _ = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            temperature=2.1,
            logprobs=True,
        )


def test_invalid_request_params_max_tokens_negative():
    with pytest.raises(ValidationError):
        _ = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            max_tokens=-1,
            logprobs=True,
        )


def test_invalid_request_params_logprobs_zero_invalid():
    with pytest.raises(ValidationError):
        _ = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            logprobs=0,  # coerces to False but should still fail via validator
        )


def test_invalid_request_params_stop_bad_type():
    with pytest.raises(ValidationError):
        _ = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            stop=123,  # invalid type
            logprobs=True,
        )


def test_generate_stream_adapter_raises_on_no_token():
    async def main():
        tok = FakeTokenizer()
        mm = FakeModelManager(tok)
        ad = FakeStrategyAdapter()
        cm = FakeClusterManager()
        mgr = InferenceManager(cm, mm, grpc_port=0, adapter=ad)
        req = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            max_tokens=1,
            logprobs=True,
        )
        agen = mgr.generate_stream(req)
        _ = await agen.__anext__()
        with pytest.raises(RuntimeError):
            _ = await agen.__anext__()  # no token queued

    asyncio.run(main())


def test_invalid_request_params_top_logprobs_range():
    with pytest.raises(ValidationError):
        _ = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            logprobs=True,
            top_logprobs=-1,
        )
    with pytest.raises(ValidationError):
        _ = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            logprobs=True,
            top_logprobs=21,
        )


def test_invalid_request_params_min_p_range():
    with pytest.raises(ValidationError):
        _ = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            min_p=-0.01,
            logprobs=True,
        )
    with pytest.raises(ValidationError):
        _ = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            min_p=1.01,
            logprobs=True,
        )


def test_invalid_request_params_min_tokens_to_keep():
    with pytest.raises(ValidationError):
        _ = ChatRequestModel(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            min_tokens_to_keep=0,
            logprobs=True,
        )
