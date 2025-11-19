import asyncio
import time
import uuid
import mlx.core as mx
import numpy as np
from typing import Optional, Any, List

from ..models import (
    ChatRequestModel, 
    ChatResponseModel, 
    ChatChoice, 
    ChatMessage, 
    ChatUsage,
    ChatCompletionReason,
    ChatLogProbs
)
from .cluster import ClusterManager
from .model_manager import ModelManager
from .strategies.base import ApiAdapterBase

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
        adapter: ApiAdapterBase
    ):
        self.cluster_manager = cluster_manager
        self.model_manager = model_manager
        self.grpc_port = grpc_port
        self.adapter = adapter
        
        self._api_callback_addr: str = ""

    async def connect_to_ring(self, first_shard_ip: str, first_shard_port: int, api_ip: str):
        await self.adapter.connect_first_shard(first_shard_ip, first_shard_port)
        self._api_callback_addr = f"{api_ip}:{self.grpc_port}"

    async def chat_completions(self, req: ChatRequestModel) -> ChatResponseModel:
        """
        Handles chat completion request.
        """
        if not self.model_manager.tokenizer:
            raise RuntimeError("Inference manager not ready (ring not connected or tokenizer not loaded)")

        tokenizer = self.model_manager.tokenizer

        try:
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
                message_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
                prompt_text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                    message_dicts,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                prompt_text = "\n".join(m.content for m in req.messages) + "\nAssistant:"
        except Exception:
            prompt_text = "\n".join(m.content for m in req.messages) + "\nAssistant:"

        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_array = mx.array(prompt_tokens)
        
        stop_id_sequences = []
        if req.stop:
            for stop_word in req.stop:
                stop_id_sequences.append(tokenizer.encode(stop_word, add_special_tokens=False))

        nonce = f"chatcmpl-{uuid.uuid4()}"
        t_start = time.perf_counter()
        t_first_token: Optional[float] = None
        tokens: List[int] = []
        token_logprobs: List[float] = []
        
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        
        completion_reason = ChatCompletionReason.LENGTH

        await self.adapter.reset_cache()

        y = prompt_array
        for _ in range(req.max_tokens):
            tok_np = y.astype(mx.int32).tolist() if hasattr(y, "tolist") else list(map(int, y))
            tok_bytes = np.array(tok_np, dtype=np.int32).tobytes(order="C")
            await self.adapter.send_tokens(nonce, tok_bytes, self._api_callback_addr)
            token = int(await self.adapter.await_token(nonce, timeout_s=300.0))

            if t_first_token is None:
                t_first_token = time.perf_counter()

            detokenizer.add_token(token)
            tokens.append(token)

            # stopping criteria
            if token == tokenizer.eos_token_id:
                completion_reason = ChatCompletionReason.STOP
                break
            y = mx.array([token], dtype=mx.int32)
                
        detokenizer.finalize()
        text = detokenizer.text
        t_end = time.perf_counter()

        metrics_dict = None
        if getattr(req, "profile", False):
            total_s = max(t_end - t_start, 1e-9)
            gen_s = max((t_end - (t_first_token or t_start)), 1e-9)
            tokens_generated = len(tokens)
            metrics_dict = {
                "total_ms": round(total_s * 1000.0, 3),
                "ttfb_ms": round(((t_first_token or t_end) - t_start) * 1000.0, 3),
                "token_gen_ms": round(gen_s * 1000.0, 3),
                "tokens_generated": tokens_generated,
                "tps_overall": round((tokens_generated / total_s) if tokens_generated else 0.0, 4),
                "tps_decoding": round((tokens_generated / gen_s) if tokens_generated else 0.0, 4),
            }

        return ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason=completion_reason,
                    message=ChatMessage(role="assistant", content=text),
                    logprobs=ChatLogProbs(
                        token_logprobs=token_logprobs,
                        top_logprobs=[],
                        tokens=tokens,
                    ),
                )
            ],
            usage=ChatUsage(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=len(tokens),
                total_tokens=len(prompt_tokens) + len(tokens),
            ),
            created=int(time.time()),
            model=req.model,
            metrics=metrics_dict,
        )

    def resolve_request(self, nonce: str, result: Any):
        self.adapter.resolve_token(nonce, int(result))
