import asyncio
import time
import uuid
import mlx.core as mx
import numpy as np
from typing import Optional, Any, List
from dnet.core.tensor import to_bytes

from .models import (
    ChatRequestModel,
    ChatResponseModel,
    ChatChoice,
    ChatMessage,
    ChatUsage,
    ChatCompletionReason,
    ChatLogProbs,
)
from .cluster import ClusterManager
from .model_manager import ModelManager
from .strategies.base import ApiAdapterBase
from dnet.core.decoding.config import DecodingConfig


async def arange(count: int):
    """Async range generator."""
    for i in range(count):
        yield i


async def azip(*async_iterables):
    """Async zip."""
    iterators = [aiter(it) for it in async_iterables]
    while True:
        try:
            results = await asyncio.gather(*[anext(it) for it in iterators])
            yield results
        except StopAsyncIteration:
            break


class InferenceManager:
    def __init__(
        self,
        cluster_manager: ClusterManager,
        model_manager: ModelManager,
        grpc_port: int,
        adapter: ApiAdapterBase,
    ):
        self.cluster_manager = cluster_manager
        self.model_manager = model_manager
        self.grpc_port = grpc_port
        self.adapter = adapter

        self._api_callback_addr: str = ""

    async def connect_to_ring(
        self, first_shard_ip: str, first_shard_port: int, api_callback_addr: str
    ) -> None:
        """
        `api_callback_addr` must be a reachable `host:port` from shards.
        For internet setups, this should be a public IP/DNS or overlay VPN IP.
        """
        await self.adapter.connect_first_shard(first_shard_ip, first_shard_port)
        self._api_callback_addr = api_callback_addr

    async def generate_stream(self, req: ChatRequestModel):
        """
        Generator for chat completion chunks.
        """
        if not self.model_manager.tokenizer:
            raise RuntimeError(
                "Inference manager not ready (ring not connected or tokenizer not loaded)"
            )

        tokenizer = self.model_manager.tokenizer

        try:
            if (
                hasattr(tokenizer, "chat_template")
                and tokenizer.chat_template is not None
            ):
                message_dicts = [
                    {"role": m.role, "content": m.content} for m in req.messages
                ]
                prompt_text = tokenizer.apply_chat_template(
                    message_dicts,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                prompt_text = (
                    "\n".join(m.content for m in req.messages) + "\nAssistant:"
                )
        except Exception:
            prompt_text = "\n".join(m.content for m in req.messages) + "\nAssistant:"

        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_array = mx.array(prompt_tokens)

        stop_id_sequences = []
        if req.stop:
            for stop_word in req.stop:
                stop_id_sequences.append(
                    tokenizer.encode(stop_word, add_special_tokens=False)
                )

        nonce = f"chatcmpl-{uuid.uuid4()}"
        t_start = time.perf_counter()
        t_first_token: Optional[float] = None
        tokens: List[int] = []

        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        last_text_len = 0

        completion_reason = ChatCompletionReason.LENGTH

        await self.adapter.reset_cache()

        # Yield initial chunk with role
        yield ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    delta=ChatMessage(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
            created=int(time.time()),
            model=req.model,
        )

        y = prompt_array
        for _ in range(req.max_tokens):
            tok_np = (
                y.astype(mx.int32)
                if hasattr(y, "astype")
                else np.array(list(map(int, y)), dtype=np.int32)
            )
            tok_bytes = to_bytes(
                tok_np,
                wire_dtype_str="int32",
                wire_mx_dtype=mx.int32,
            )

            decoding_config = DecodingConfig(
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                min_p=req.min_p if hasattr(req, "min_p") else 0.0,
                min_tokens_to_keep=req.min_tokens_to_keep
                if hasattr(req, "min_tokens_to_keep")
                else 1,
            )

            # Send tokens to first shard
            await self.adapter.send_tokens(
                tokens=tok_bytes,
                nonce=nonce,
                callback_addr=self._api_callback_addr,
                logprobs=req.logprobs if req.logprobs else False,
                top_logprobs=req.top_logprobs if req.top_logprobs else 0,
                decoding_config=decoding_config,
            )
            result = await self.adapter.await_token(nonce, timeout_s=300.0)
            token = int(result.token_id)

            # Accumulate logprobs
            token_logprobs = []
            top_logprobs_list = []
            if req.logprobs:
                token_logprobs.append(result.logprob)
                top_logprobs_list.append(result.top_logprobs)

            if t_first_token is None:
                t_first_token = time.perf_counter()

            detokenizer.add_token(token)
            tokens.append(token)

            # mlx_lm detokenizer usually updates .text property
            full_text = detokenizer.text
            delta_text = full_text[last_text_len:]
            last_text_len = len(full_text)

            # Yield chunk
            yield ChatResponseModel(
                id=nonce,
                choices=[
                    ChatChoice(
                        index=0,
                        delta=ChatMessage(role="assistant", content=delta_text),
                        logprobs=ChatLogProbs(
                            token_logprobs=token_logprobs,
                            top_logprobs=top_logprobs_list,
                            tokens=[token],
                        )
                        if req.logprobs
                        else None,
                        finish_reason=None,
                    )
                ],
                created=int(time.time()),
                model=req.model,
            )

            # stopping criteria
            if token == tokenizer.eos_token_id:
                completion_reason = ChatCompletionReason.STOP
                break
            y = mx.array([token], dtype=mx.int32)

        detokenizer.finalize()

        metrics_dict = None
        t_end = time.perf_counter()
        if getattr(req, "profile", False):
            total_s = max(t_end - t_start, 1e-9)
            gen_s = max((t_end - (t_first_token or t_start)), 1e-9)
            tokens_generated = len(tokens)
            metrics_dict = {
                "total_ms": round(total_s * 1000.0, 3),
                "ttfb_ms": round(((t_first_token or t_end) - t_start) * 1000.0, 3),
                "token_gen_ms": round(gen_s * 1000.0, 3),
                "tokens_generated": tokens_generated,
                "tps_overall": round(
                    (tokens_generated / total_s) if tokens_generated else 0.0, 4
                ),
                "tps_decoding": round(
                    (tokens_generated / gen_s) if tokens_generated else 0.0, 4
                ),
            }

        # Final chunk with finish reason
        yield ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    delta=ChatMessage(role="assistant", content=""),
                    finish_reason=completion_reason,
                )
            ],
            created=int(time.time()),
            model=req.model,
            metrics=metrics_dict,
            usage=ChatUsage(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=len(tokens),
                total_tokens=len(prompt_tokens) + len(tokens),
            ),
        )

    async def chat_completions(self, req: ChatRequestModel) -> ChatResponseModel:
        """
        Handles chat completion request (non-streaming).
        """
        full_content = ""
        tokens = []
        token_logprobs = []
        top_logprobs_list = []
        completion_reason = ChatCompletionReason.LENGTH
        nonce = ""
        metrics_dict = None
        usage = None

        async for chunk in self.generate_stream(req):
            nonce = chunk.id
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                full_content += choice.delta.content

            if choice.logprobs:
                if choice.logprobs.token_logprobs:
                    token_logprobs.extend(choice.logprobs.token_logprobs)
                if choice.logprobs.top_logprobs:
                    top_logprobs_list.extend(choice.logprobs.top_logprobs)
                if choice.logprobs.tokens:
                    tokens.extend(choice.logprobs.tokens)

            if choice.finish_reason:
                completion_reason = choice.finish_reason

            if chunk.metrics:
                metrics_dict = chunk.metrics

            if chunk.usage:
                usage = chunk.usage

        return ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason=completion_reason,
                    message=ChatMessage(role="assistant", content=full_content),
                    logprobs=ChatLogProbs(
                        token_logprobs=token_logprobs,
                        top_logprobs=top_logprobs_list,
                        tokens=tokens,
                    )
                    if req.logprobs
                    else None,
                )
            ],
            usage=usage,
            created=int(time.time()),
            model=req.model,
            metrics=metrics_dict,
        )

    def resolve_request(self, nonce: str, result: Any):
        self.adapter.resolve_token(nonce, result)
