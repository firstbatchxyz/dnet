from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Optional, Any
from urllib.parse import urlparse

import grpc
from grpc import aio as aio_grpc
import numpy as np
from mlx.core import Dtype

from dnet.ring.memory_pool import LayerAwareMemoryPool
from ...utils.grpc_config import GRPC_AIO_OPTIONS
from ...utils.logger import logger
from ...utils.time import utc_epoch_now
from ...utils.serialization import dtype_map, tensor_to_bytes
from ...utils.model import ModelMetadata
from ...protos import shard_api_comm_pb2, shard_api_comm_pb2_grpc, dnet_ring_pb2
from ..data_types import ActivationMessage


class SendMixin:
    next_node_stub: Optional[Any]
    activation_computed_queue: asyncio.Queue[ActivationMessage]
    node_id: int
    _profile: bool
    output_pool: LayerAwareMemoryPool
    running: bool
    model_metadata: ModelMetadata
    _prefetch_pending: set[int]
    _prefetch_active = 0
    weight_prefetch_queue: asyncio.Queue[int]
    _wire_dtype_str: str
    _wire_mx_dtype: Dtype
    _assigned_set: set[int]
    window_size: int

    _prefetch_to_ram: Callable[[int], None]
    _enqueue_weight_prefetch: Callable[[int], None]
    _prefetch_pause: asyncio.Event
    _next_local_layers: Callable[[int, int], list[int]]
    _clear_prefetch_state: Callable[[], None]

    @dataclass
    class _StreamCtx:
        nonce: str
        queue: asyncio.Queue
        call: Optional[Any] = None
        ack_task: Optional[asyncio.Task] = None
        open: bool = False
        disabled: bool = False
        disabled_until: float = 0.0
        last_seq: int = 0
        last_activity_t: float = 0.0

    async def _ensure_stream(self, nonce: str):
        try:
            if not getattr(self, "_streaming_enabled", False):
                return None
            if self.next_node_stub is None:
                return None
            # Ensure the stub supports streaming
            if not hasattr(self.next_node_stub, "StreamActivations"):
                self._streaming_enabled = False
                return None
            ctx = getattr(self, "_streams", {}).get(nonce)
            if ctx and ctx.open:
                # If the stream was temporarily disabled (e.g., backpressure),
                # automatically re-enable after the backoff interval elapses.
                try:
                    loop = asyncio.get_running_loop()
                    if ctx.disabled and loop.time() >= ctx.disabled_until:
                        ctx.disabled = False
                except Exception:
                    pass
                return ctx

            ctx = SendMixin._StreamCtx(nonce=nonce, queue=asyncio.Queue(maxsize=64))
            if not hasattr(self, "_streams"):
                self._streams = {}
            self._streams[nonce] = ctx

            async def _req_iter():
                while True:
                    item = await ctx.queue.get()
                    if item is None:
                        break
                    yield item

            call = self.next_node_stub.StreamActivations(_req_iter())  # type: ignore[attr-defined]
            ctx.call = call
            ctx.open = True
            ctx.last_activity_t = asyncio.get_running_loop().time()

            async def _ack_reader():
                try:
                    async for ack in call:
                        if not getattr(ack, "accepted", True):
                            logger.debug(
                                "[STREAM][ACK] nonce=%s seq=%s accepted=0 msg=%s",
                                getattr(ack, "nonce", ""),
                                getattr(ack, "seq", -1),
                                getattr(ack, "message", ""),
                            )
                        # Temporary backoff on backpressure; do not permanently disable
                        msg = str(getattr(ack, "message", "")).lower()
                        if "backpressure" in msg:
                            backoff_s = float(getattr(self, "_stream_backoff_s", 0.5))
                            loop = asyncio.get_running_loop()
                            ctx.disabled = True
                            ctx.disabled_until = loop.time() + backoff_s
                except Exception as e:
                    logger.error("[STREAM] ack reader error: %s", e)
                    ctx.open = False
                    ctx.disabled = True

            ctx.ack_task = asyncio.create_task(_ack_reader())
            return ctx
        except Exception as e:
            logger.warning("_ensure_stream failed: %s", e)
            return None

    async def _end_stream(self, nonce: str, *, eor: bool = False):
        ctx = getattr(self, "_streams", {}).pop(nonce, None)
        if not ctx:
            return

        if eor and ctx.open and not ctx.disabled:
            try:
                ctx.last_seq += 1
                await ctx.queue.put(
                    dnet_ring_pb2.ActivationFrame(
                        request=dnet_ring_pb2.ActivationRequest(nonce=nonce),
                        seq=ctx.last_seq,
                        end_of_request=True,
                    )
                )
            except Exception:
                pass
        try:
            await ctx.queue.put(None)  # close iterator
        except Exception:
            pass
        if ctx.ack_task:
            ctx.ack_task.cancel()

    async def _stream_sweeper(self):
        idle_s = float(getattr(self, "_stream_idle_s", 2.0))
        while getattr(self, "running", False):
            try:
                if not getattr(self, "_streaming_enabled", False):
                    await asyncio.sleep(1.0)
                    continue
                now = asyncio.get_running_loop().time()
                for nonce, ctx in list(getattr(self, "_streams", {}).items()):
                    if (now - ctx.last_activity_t) > idle_s:
                        await self._end_stream(nonce, eor=False)
                await asyncio.sleep(1.0)
            except Exception:
                await asyncio.sleep(1.0)

    async def _forward_activation(self, request: dnet_ring_pb2.ActivationRequest):
        try:
            t0 = time.perf_counter()
            response = await self.next_node_stub.SendActivation(request)  # type: ignore
            rpc_ms = (time.perf_counter() - t0) * 1000.0
            if not response.success:
                logger.warning("Next node reported error: %s", response.message)
            logger.info(
                "[PROFILE][TX] node=%s nonce=%s forwarded_layer=%s rpc_ms=%.2f",
                self.node_id,
                request.nonce,
                request.activation.layer_id + 1,
                rpc_ms,
            )
        except Exception as e:
            logger.error("Failed to forward activation: %s", e)

    async def _send_worker(self):
        while self.running or (
            hasattr(self, "activation_computed_queue")
            and not self.activation_computed_queue.empty()
        ):
            try:
                activation_msg = await self.activation_computed_queue.get()
                if getattr(activation_msg, "tx_enq_perf_t", 0.0) and self._profile:
                    q_wait_ms = (
                        time.perf_counter() - activation_msg.tx_enq_perf_t
                    ) * 1000.0
                    logger.info(
                        "[PROFILE][QUEUE-TX] node=%s nonce=%s wait_ms=%.3f size=%s",
                        self.node_id,
                        activation_msg.nonce,
                        q_wait_ms,
                        self.activation_computed_queue.qsize(),
                    )
                await self._send_activation(activation_msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Send worker error: %s", e)

    async def _send_activation(self, activation_msg: ActivationMessage):
        if not self._check_model_loaded() or not self.output_pool:
            logger.error(
                "Node %s: Cannot send activation - model not loaded", self.node_id
            )
            return
        try:
            # Handle final token path (end-shard sampling)
            if getattr(activation_msg, "is_final", False):
                cb = activation_msg.callback_url or ""
                parsed = urlparse(cb) if cb else None
                t_rpc = time.perf_counter()
                if parsed and parsed.scheme == "grpc":
                    addr = parsed.netloc
                    if not addr:
                        logger.error("Invalid gRPC callback URL for token: %s", cb)
                        return
                    # Ensure API channel/stub
                    if (getattr(self, "api_channel", None) is None) or (
                        addr != getattr(self, "api_address", None)
                    ):
                        try:
                            if getattr(self, "api_channel", None) is not None:
                                await self.api_channel.close()
                        except Exception:
                            pass
                        self.api_address = addr
                        self.api_channel = aio_grpc.insecure_channel(
                            addr, options=GRPC_AIO_OPTIONS
                        )
                        self.api_stub = shard_api_comm_pb2_grpc.ShardApiServiceStub(
                            self.api_channel
                        )
                    try:
                        req = shard_api_comm_pb2.TokenRequest(
                            nonce=activation_msg.nonce,
                            token_id=int(getattr(activation_msg, "token_id", -1)),
                            timestamp=utc_epoch_now(),
                        )
                        resp = await self.api_stub.SendToken(req)  # type: ignore
                        rpc_ms = (time.perf_counter() - t_rpc) * 1000.0
                        if not resp.success:
                            logger.error(
                                "API SendToken failed for %s: %s",
                                activation_msg.nonce,
                                resp.message,
                            )
                        if self._profile:
                            logger.info(
                                "[PROFILE][TX-TOKEN][gRPC] node=%s nonce=%s token=%s rpc_ms=%.2f",
                                self.node_id,
                                activation_msg.nonce,
                                int(getattr(activation_msg, "token_id", -1)),
                                rpc_ms,
                            )
                    except Exception as e:
                        logger.exception("Error sending token via gRPC: %s", e)
                else:
                    logger.error(
                        "No valid gRPC callback for token; cannot deliver nonce=%s",
                        activation_msg.nonce,
                    )
                return

            used_pool = False
            shaped = getattr(activation_msg, "tensor", None)
            if shaped is None:
                output_buffer = self.output_pool.get_buffer(activation_msg.pool_id)
                if output_buffer is None:
                    logger.error(
                        "Failed to get output buffer %s", activation_msg.pool_id
                    )
                    return
                data_size = int(np.prod(activation_msg.shape))
                shaped = output_buffer[:data_size].reshape(activation_msg.shape)
                used_pool = True

            if self._profile:
                logger.info(
                    "[PROFILE][SER-START] node=%s nonce=%s prefetch_active=%s pending=%s queue=%s",
                    self.node_id,
                    activation_msg.nonce,
                    self._prefetch_active,
                    len(self._prefetch_pending),
                    self.weight_prefetch_queue.qsize(),
                )
            t_ser = time.perf_counter()
            t_cast = t_ser

            _len_bytes = int(getattr(shaped, "nbytes", 0))
            _do_compress = bool(
                getattr(self, "_compress", False)
                and _len_bytes >= getattr(self, "_compress_min_bytes", 65536)
            )
            if _do_compress:
                # Skip compression for decode.
                _do_compress = False
            try:
                wire_np_dtype = dtype_map[self._wire_dtype_str]
            except Exception:
                wire_np_dtype = np.float16  # reasonable default fallback
            if isinstance(shaped, np.ndarray):
                if shaped.dtype != wire_np_dtype:
                    shaped = shaped.astype(wire_np_dtype, copy=False)
            else:
                # MLX array -> cast to desired wire dtype
                if str(shaped.dtype) != self._wire_dtype_str:
                    shaped = shaped.astype(self._wire_mx_dtype)
            activation_msg.dtype = self._wire_dtype_str
            t_cast = time.perf_counter()

            if isinstance(shaped, np.ndarray):
                data = shaped.tobytes(order="C")
            else:
                data = tensor_to_bytes(shaped)

            ser_ms = (time.perf_counter() - t_ser) * 1000.0
            cast_ms = (t_cast - t_ser) * 1000.0

            nxt = activation_msg.layer_id + 1
            if (nxt < self.model_metadata.num_layers) and (
                nxt not in self._assigned_set
            ):
                if self.next_node_stub:
                    request = activation_msg.to_proto(data)
                    request.timestamp = utc_epoch_now()
                    # Prefer streaming if enabled/available; fallback to unary
                    stream_used = False
                    ctx = await self._ensure_stream(activation_msg.nonce)
                    if (
                        ctx
                        and ctx.open
                        and not ctx.disabled
                        and hasattr(dnet_ring_pb2, "ActivationFrame")
                    ):
                        try:
                            ctx.last_seq += 1
                            frame = dnet_ring_pb2.ActivationFrame(
                                request=request,
                                seq=ctx.last_seq,
                                end_of_request=False,
                            )
                            await ctx.queue.put(frame)
                            ctx.last_activity_t = asyncio.get_running_loop().time()
                            stream_used = True
                            if self._profile:
                                logger.info(
                                    "[PROFILE][STREAM-ENQ] nonce=%s seq=%s q=%s",
                                    activation_msg.nonce,
                                    ctx.last_seq,
                                    ctx.queue.qsize(),
                                )
                        except Exception as e:
                            logger.warning(
                                "[STREAM] enqueue failed; fallback to unary: %s", e
                            )
                            ctx.disabled = True

                    if not stream_used:
                        t0 = time.perf_counter()
                        max_attempts = max(1, int(getattr(self, "_send_retries", 3)))
                        last_exc: Optional[Exception] = None
                        for attempt in range(1, max_attempts + 1):
                            try:
                                # FIXME: use response here?
                                _ = await self.next_node_stub.SendActivation(
                                    request, timeout=30.0
                                )  # type: ignore
                                break
                            except grpc.aio.AioRpcError as e:  # type: ignore
                                last_exc = e
                                code = e.code()
                                if code in {
                                    grpc.StatusCode.UNAVAILABLE,
                                    grpc.StatusCode.CANCELLED,
                                    grpc.StatusCode.DEADLINE_EXCEEDED,
                                }:
                                    logger.warning(
                                        "SendActivation attempt %s/%s failed (%s); reconnecting to %s",
                                        attempt,
                                        max_attempts,
                                        code.name,
                                        self.next_node_address,  # FIXME: will be `next_node` !
                                    )
                                    await self._reconnect_next_node()  # FIXME: !!!
                                    await asyncio.sleep(min(0.25 * attempt, 1.0))
                                    continue
                                raise
                        else:
                            raise last_exc  # type: ignore
                        rpc_ms = (time.perf_counter() - t0) * 1000.0
                        logger.info(
                            "[PROFILE][TX] node=%s nonce=%s next_layer=%s payload_kb=%.1f serialize_ms=%.3f rpc_ms=%.2f cast_ms=%.3f",
                            self.node_id,
                            activation_msg.nonce,
                            activation_msg.layer_id + 1,
                            (len(data) / 1024),
                            ser_ms,
                            rpc_ms,
                            cast_ms,
                        )

                    try:
                        t_flush = time.perf_counter()
                        flushed = 0
                        for lid in list(self._prefetch_pending):
                            self._prefetch_to_ram(lid)
                            self.weight_prefetch_queue.put_nowait(lid)
                            self._prefetch_pending.discard(lid)
                            flushed += 1
                        if self._profile:
                            logger.info(
                                "[PROFILE][SER-END] node=%s nonce=%s flushed=%s flush_ms=%.2f",
                                self.node_id,
                                activation_msg.nonce,
                                flushed,
                                (time.perf_counter() - t_flush) * 1000.0,
                            )
                    except Exception:
                        pass
                    finally:
                        try:
                            self._prefetch_pause.clear()
                        except Exception:
                            pass

                    try:
                        next_window = self._next_local_layers(
                            activation_msg.layer_id, self.window_size
                        )
                        for nl in next_window:
                            self._prefetch_to_ram(nl)
                            self._enqueue_weight_prefetch(nl)
                    except Exception:
                        pass
                else:
                    logger.error(
                        "Cannot forward activation - no next node configured; end shard should sample inline."
                    )
            else:
                logger.error(
                    "Final activation reached send path unexpectedly; sampling should occur on end shard."
                )

                # Resume prefetch and flush deferred
                try:
                    t_flush = time.perf_counter()
                    flushed = 0
                    for lid in list(self._prefetch_pending):
                        self._prefetch_to_ram(lid)
                        self.weight_prefetch_queue.put_nowait(lid)
                        self._prefetch_pending.discard(lid)
                        flushed += 1
                    if self._profile:
                        logger.info(
                            "[PROFILE][SER-END] node=%s nonce=%s flushed=%s flush_ms=%.2f",
                            self.node_id,
                            activation_msg.nonce,
                            flushed,
                            (time.perf_counter() - t_flush) * 1000.0,
                        )
                except Exception:
                    pass
                finally:
                    try:
                        self._prefetch_pause.clear()
                    except Exception:
                        pass

                # Clear scheduling at request end
                try:
                    self._clear_prefetch_state()
                except Exception:
                    pass

                # Optional: explicitly end the per-nonce stream on request completion
                # Enable by setting RING_EXPLICIT_EOR=1 when you emit a true end-of-request signal.
                try:
                    if getattr(self, "_explicit_eor", False):
                        if (
                            getattr(self, "_streams", None)
                            and activation_msg.nonce in self._streams
                        ):
                            await self._end_stream(activation_msg.nonce, eor=True)
                except Exception:
                    pass

            # Release resources at end of send
            try:
                activation_msg.tensor = None
            except Exception:
                pass
            if used_pool:
                try:
                    self.output_pool.release(activation_msg.pool_id)
                except Exception:
                    pass
        except Exception as e:
            logger.exception("Error sending activation: %s", e)
