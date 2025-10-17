"""API utilities for ring topology generation."""

import asyncio
import time
from typing import AsyncGenerator, Dict, Tuple

import mlx.core as mx
import numpy as np

from ...protos import dnet_ring_pb2
from ...protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from ...utils.logger import logger
from .models import ChatBaseParams


def utc_epoch_now() -> int:
    """Return current UTC timestamp in milliseconds."""
    return int(time.time() * 1000)


def create_generate_step_for_ring_with_grpc(
    stub: DnetRingServiceStub,
    *,
    callback_protocol: str = "http",
    callback_addr: str | None = None,
    compression: float = 0.0,
):
    pb2 = dnet_ring_pb2

    async def generate_step(
        nonce: str,
        node_origin: str,
        prompt: mx.array,
        pending_requests: Dict[str, asyncio.Future],
        params: ChatBaseParams,
    ) -> AsyncGenerator[Tuple[mx.array, mx.array], None]:
        repetition_penalty = params.repetition_penalty
        repetition_context_size = params.repetition_context_size or 20

        reset_response = await stub.ResetCache(pb2.ResetCacheRequest())  # type: ignore
        logger.info("ResetCache Response: %s", reset_response.message)

        y = prompt
        repetition_context = prompt.tolist()
        if repetition_context_size:
            repetition_context = repetition_context[-repetition_context_size:]  # type: ignore

        async def _step(y):
            nonlocal repetition_context

            future = asyncio.get_running_loop().create_future()
            # Record API-side monotonic start time for end-to-end latency
            try:
                future._api_t0 = asyncio.get_running_loop().time()  # type: ignore[attr-defined]
            except Exception:
                pass
            pending_requests[nonce] = future

            # send the activation to next shards
            # Build callback target
            if callback_protocol == "grpc" and callback_addr:
                callback_url = f"grpc://{callback_addr}"
            else:
                callback_url = f"http://{node_origin}/results/{nonce}"

            # Build token payload
            tok_np = np.array(y, dtype=np.int32)
            act = pb2.Activation(
                batch_size=1,
                shape=[int(tok_np.size)],
                dtype="tokens",
                layer_id=-1,
                data=tok_np.tobytes(order="C"),
            )

            activation_message = pb2.ActivationRequest(
                nonce=nonce,
                node_origin=node_origin,
                timestamp=utc_epoch_now(),
                activation=act,
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

            # Wait for token callback from last shard
            try:
                result = await asyncio.wait_for(future, timeout=3000.0)
            except asyncio.TimeoutError as e:
                raise RuntimeError(
                    "Did not receive token corresponding to %s, err: %s", nonce, e
                )
            except Exception as e:
                raise RuntimeError(f"Request {nonce} failed with exception {e!r}")
            finally:
                pending_requests.pop(nonce, None)

            # Only token IDs are supported (end shard samples inline)
            if not isinstance(result, int):
                raise RuntimeError(
                    "Expected token callback from shard, but received activation payload"
                )
            y = mx.array([int(result)], dtype=mx.int32)
            logprobs = mx.array([], dtype=mx.float32)
            if repetition_penalty:
                repetition_context.append(y.item())
            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:
                    repetition_context = repetition_context[-repetition_context_size:]
            return y, logprobs

        y, logprobs = await _step(y)
        mx.async_eval(y)
        while True:
            next_y, next_logprobs = await _step(y)
            mx.async_eval(next_y)
            yield y.item(), logprobs
            y, logprobs = next_y, next_logprobs

    return generate_step
