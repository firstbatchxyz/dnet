"""API utilities for ring topology generation."""

import asyncio
import time
from typing import AsyncGenerator, Dict, Optional, Tuple, cast

import mlx.core as mx
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from ..model import BaseRingModel
from ...protos import dnet_ring_pb2
from ...protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from ...utils.async_utils import make_cache
from ...utils.logger import logger
from ...utils.serialization import bytes_to_tensor, tensor_to_bytes
from .models import ChatBaseParams, RecieveResultRequest


def utc_epoch_now() -> int:
    """Return current UTC timestamp in milliseconds."""
    return int(time.time() * 1000)


def create_generate_step_for_ring_with_grpc(
    stub: DnetRingServiceStub,
    *,
    callback_protocol: str = "http",
    callback_addr: Optional[str] = None,
):
    """Create a generation step function for ring topology with gRPC.

    Args:
        stub: gRPC stub for communicating with first shard
        callback_protocol: Callback protocol ("http" or "grpc")
        callback_addr: Callback address for gRPC callbacks

    Returns:
        Generate step async function
    """
    pb2 = dnet_ring_pb2

    async def generate_step(
        nonce: str,
        node_origin: str,
        prompt: mx.array,
        model: BaseRingModel,
        pending_requests: Dict[str, asyncio.Future],
        params: ChatBaseParams,
    ) -> AsyncGenerator[Tuple[mx.array, mx.array], None]:
        temp = params.temperature or 0.0
        repetition_penalty = params.repetition_penalty
        repetition_context_size = params.repetition_context_size or 20
        top_p = params.top_p or 1.0
        logit_bias = params.logit_bias

        reset_response = await stub.ResetCache(pb2.ResetCacheRequest())  # type: ignore
        logger.info(f"ResetCache Response: {reset_response.message}")

        # Create sampler and logits processors for mlx-lm 0.24.0
        sampler = make_sampler(temp or 0.0, top_p, 0.0, 1)
        logits_processors = make_logits_processors(
            logit_bias, repetition_penalty, repetition_context_size
        )

        def sample(logits: mx.array, context=None) -> Tuple[mx.array, mx.array]:
            if logits_processors and context is not None:
                for proc in logits_processors:
                    logits = proc(context, logits)

            # Ensure logits is 2D for sampling (batch_size, vocab_size)
            if logits.ndim == 1:
                logits = logits[None, :]  # Add batch dimension

            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            token = sampler(logprobs)

            # Return 1D logprobs for the selected position
            if logprobs.ndim == 2 and logprobs.shape[0] == 1:
                logprobs = logprobs[0]  # Remove batch dimension

            return token, logprobs

        y = prompt
        _model_cache = make_cache(model)

        repetition_context = prompt.tolist()
        if repetition_context_size:
            repetition_context = repetition_context[-repetition_context_size:]  # type: ignore

        async def _step(y):
            nonlocal repetition_context

            # compute embedding
            t_emb = asyncio.get_running_loop().time()
            output = model.embed(y[None])
            if output.dtype == mx.bfloat16:
                output = output.astype(mx.float16)
            emb_ms = (asyncio.get_running_loop().time() - t_emb) * 1000.0
            logger.info(f"[PROFILE][API] nonce={nonce} embed_ms={emb_ms:.2f}")

            future = asyncio.get_running_loop().create_future()
            pending_requests[nonce] = future

            # send the activation to next shards
            # Build callback target
            if callback_protocol == "grpc" and callback_addr:
                callback_url = f"grpc://{callback_addr}"
            else:
                callback_url = f"http://{node_origin}/results/{nonce}"

            activation_message = pb2.ActivationRequest(
                nonce=nonce,
                node_origin=node_origin,
                timestamp=utc_epoch_now(),
                activation=pb2.Activation(
                    batch_size=1,  # TODO handle non-unit cases
                    shape=cast(Tuple[int, ...], output.shape),
                    dtype=str(output.dtype),
                    layer_id=-1,
                    data=tensor_to_bytes(output),
                ),
                callback_url=callback_url,
            )
            t_rpc = asyncio.get_running_loop().time()
            response = await stub.SendActivation(activation_message)  # type: ignore
            api_rpc_ms = (asyncio.get_running_loop().time() - t_rpc) * 1000.0
            logger.info(
                f"[PROFILE][API-TX] nonce={nonce} to=shard0 rpc_ms={api_rpc_ms:.2f} payload_kb={(len(activation_message.activation.data) / 1024):.1f}"
            )
            if not response.success:
                raise RuntimeError(
                    f"Sending activation {nonce} to {node_origin} was not succesful"
                )

            try:
                # Wait for HTTP callback from N2
                result = await asyncio.wait_for(future, timeout=3000.0)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Did not receive activation corresponding to {nonce}"
                )
            except Exception as e:
                raise RuntimeError(f"Request {nonce} failed with exception {e!r}")
            finally:
                pending_requests.pop(nonce, None)

            # Apply final RMSNorm before projecting to vocab logits
            t_head = asyncio.get_running_loop().time()
            output = bytes_to_tensor(
                RecieveResultRequest.decode(result.data), result.dtype
            )
            output = output.reshape(result.shape)
            output = model.normalize(output)
            output = model.lm_project(output)
            head_ms = (asyncio.get_running_loop().time() - t_head) * 1000.0
            logger.info(f"[PROFILE][API] nonce={nonce} final_head_ms={head_ms:.2f}")

            # sample next token - ensure API receives 1D logprobs
            # Ensure output is 3D (batch, seq, vocab)
            if output.ndim == 2:
                output = output[None, :, :]  # Add batch dimension

            # Extract logits for last token position, keep batch dim for sampler
            logits_2d = output[:, -1, :]  # Shape: (1, vocab_size)

            # Apply repetition penalty if needed
            if logits_processors and repetition_penalty:
                for proc in logits_processors:
                    logits_2d = proc(repetition_context, logits_2d)

            # Compute log probabilities (2D), and take 1D view for API indexing
            logprobs_2d = logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)
            logprobs = logprobs_2d[0]

            # Sample next token from 2D logprobs
            y = sampler(logprobs_2d)
            if repetition_penalty:
                repetition_context.append(y.item())  # type: ignore
            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:  # type: ignore
                    repetition_context = repetition_context[  # type: ignore
                        -repetition_context_size:
                    ]
            return y, logprobs  # logprobs already has correct shape from sample()

        y, logprobs = await _step(y)
        mx.async_eval(y)
        while True:
            next_y, next_logprobs = await _step(y)
            mx.async_eval(next_y)
            yield y.item(), logprobs  # type: ignore
            y, logprobs = next_y, next_logprobs

    return generate_step
