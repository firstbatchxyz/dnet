from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Any
from urllib.parse import urlparse

import grpc
from grpc import aio as aio_grpc
import numpy as np

from dnet_p2p import (
    DnetDeviceProperties,
    discover_thunderbolt_connection,
    ThunderboltConnection,
)
from dnet.utils.latency import DeviceLatencyResult, LatencyMeasurement, LatencyResults

from ...utils.grpc_config import GRPC_AIO_OPTIONS
from ...utils.logger import logger
from ...utils.time import utc_epoch_now
from ...utils.serialization import dtype_map, tensor_to_bytes
from ...protos import shard_api_comm_pb2, shard_api_comm_pb2_grpc, dnet_ring_pb2
from ..data_types import ActivationMessage

from .attrib import RingShardNodeAttributes


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


class CommsMixin(RingShardNodeAttributes):
    """Communication-related methods for ring shard node."""

    async def _ensure_stream(self, nonce: str):
        try:
            if not self._streaming_enabled:
                return None
            if self.next_node_stub is None:
                return None
            # Ensure the stub supports streaming
            if not hasattr(self.next_node_stub, "StreamActivations"):
                self._streaming_enabled = False
                return None
            ctx = self._streams.get(nonce)
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

            ctx = _StreamCtx(nonce=nonce, queue=asyncio.Queue(maxsize=64))
            # _streams exists; initialized in node __init__
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
                            backoff_s = float(self._stream_backoff_s)
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
        ctx = self._streams.pop(nonce, None)
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
        idle_s = float(self._stream_idle_s)
        while self.running:
            try:
                if not self._streaming_enabled:
                    await asyncio.sleep(1.0)
                    continue
                now = asyncio.get_running_loop().time()
                for nonce, ctx in list(self._streams.items()):
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
                if activation_msg.tx_enq_perf_t and self._profile:
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

    async def _send_token_worker(self):
        """Dedicated worker for final token delivery to API.

        Separating this from ring-forward sends avoids head-of-line blocking
        when ring transport experiences retries/timeouts.
        """
        while self.running or (
            hasattr(self, "activation_token_queue")
            and not self.activation_token_queue.empty()
        ):
            try:
                activation_msg = await self.activation_token_queue.get()
                if activation_msg.tx_enq_perf_t and self._profile:
                    q_wait_ms = (
                        time.perf_counter() - activation_msg.tx_enq_perf_t
                    ) * 1000.0
                    logger.info(
                        "[PROFILE][QUEUE-TX] node=%s nonce=%s wait_ms=%.3f size=%s",
                        self.node_id,
                        activation_msg.nonce,
                        q_wait_ms,
                        self.activation_token_queue.qsize(),
                    )
                await self._send_activation(activation_msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Token send worker error: %s", e)

    async def _send_activation(self, activation_msg: ActivationMessage):
        if not self.output_pool or not self.model_metadata:
            logger.error(
                "Node %s: Cannot send activation - output pool / model metadata not initialized",
                self.node_id,
            )
            return
        try:
            if activation_msg.is_final:
                try:
                    if self._mode == "offload" and self.window_size > 0:
                        first_window = self._assigned_sorted[: self.window_size]
                        if first_window:
                            loop = asyncio.get_running_loop()
                            fut = loop.run_in_executor(
                                self.executor,
                                self._prepare_window_blocking,
                                list(first_window),
                            )
                            self._prepared_by_nonce[activation_msg.nonce] = (
                                list(first_window),
                                fut,
                            )
                except Exception:
                    pass
                cb = activation_msg.callback_url or ""
                parsed = urlparse(cb) if cb else None
                t_rpc = time.perf_counter()
                if parsed and parsed.scheme == "grpc":
                    addr = parsed.netloc
                    if not addr:
                        logger.error("Invalid gRPC callback URL for token: %s", cb)
                        return
                    # Ensure API channel/stub
                    if (self.api_channel is None) or (addr != self.api_address):
                        # close existing channel if any
                        try:
                            if self.api_channel is not None:
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

            # FIXME: shaped var is a bit weird (is it np_array or mlx_array), @andthattoo shall check
            shaped = activation_msg.tensor
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
                    "[PROFILE][SER-START] node=%s nonce=%s",
                    self.node_id,
                    activation_msg.nonce,
                )
            t_ser = time.perf_counter()
            t_cast = t_ser

            _len_bytes = int(getattr(shaped, "nbytes", 0))
            _do_compress = bool(
                self._compress and _len_bytes >= self._compress_min_bytes
            )
            if _do_compress:
                # Skip compression for decode.
                _do_compress = False
            try:
                wire_np_dtype = dtype_map[self._wire_dtype_str]
            except Exception:
                wire_np_dtype = np.float16  # reasonable default fallback

            if isinstance(shaped, np.ndarray):
                logger.warning("Activation tensor is a numpy array!!!")
                if shaped.dtype != wire_np_dtype:
                    # FIXME: numpy vs mx array here
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
                    if self._mode == "offload" and self.window_size > 0:
                        next_window = self._next_local_layers(
                            activation_msg.layer_id, self.window_size
                        )
                        loop = asyncio.get_running_loop()
                        if next_window:
                            fut = loop.run_in_executor(
                                self.executor,
                                self._prepare_window_blocking,
                                list(next_window),
                            )
                            self._prepared_by_nonce[activation_msg.nonce] = (
                                list(next_window),
                                fut,
                            )
                        else:
                            first_window = self._assigned_sorted[: self.window_size]
                            if first_window:
                                fut = loop.run_in_executor(
                                    self.executor,
                                    self._prepare_window_blocking,
                                    list(first_window),
                                )
                                self._prepared_by_nonce[activation_msg.nonce] = (
                                    list(first_window),
                                    fut,
                                )
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
                        # In fit mode, avoid long unary stalls: use short deadline and min retries
                        # Streaming should be the norm; unary is a quick safety valve only.
                        ring_timeout = 3.0 if self._mode == "fit" else 30.0
                        ring_retries = (
                            1
                            if self._mode == "fit"
                            else max(1, int(self._send_retries))
                        )
                        # Emit a clear fallback log with reason/context
                        if self._profile:
                            if ctx is None:
                                reason = "no_stream_ctx"
                            elif not ctx.open:
                                reason = "stream_closed"
                            elif ctx.disabled:
                                reason = "stream_disabled"
                            else:
                                reason = "enqueue_failed"
                            logger.warning(
                                "[STREAM->UNARY] node=%s nonce=%s reason=%s mode=%s timeout_s=%.1f retries=%d",
                                self.node_id,
                                activation_msg.nonce,
                                reason,
                                self._mode,
                                ring_timeout,
                                ring_retries,
                            )
                        t0 = time.perf_counter()
                        last_exc: Optional[Exception] = None
                        for attempt in range(1, ring_retries + 1):
                            try:
                                # FIXME: use response here?
                                _ = await self.next_node_stub.SendActivation(
                                    request, timeout=ring_timeout
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
                                        "SendActivation attempt %s/%s failed (%s); reconnecting...",
                                        attempt,
                                        ring_retries,
                                        code.name,
                                    )
                                    await self._reconnect_next_node()
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
                else:
                    logger.error(
                        "Cannot forward activation - no next node configured; end shard should sample inline."
                    )
            else:
                logger.error(
                    "Final activation reached send path unexpectedly; sampling should occur on end shard."
                )

                # Clear scheduling at request end
                # Sequential offload: prefetch state is unused

                # Optional: explicitly end the per-nonce stream on request completion
                # Enable by setting RING_EXPLICIT_EOR=1 when you emit a true end-of-request signal.
                try:
                    if self._explicit_eor:
                        if (
                            hasattr(self, "_streams")
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

    async def _connect_next_node(self) -> bool:
        """Connect to next node in ring.

        Returns:
            True if connected or no next node, False on failure
        """
        if not self.next_node:
            logger.info(f"Shard node {self.node_id} is the final shard (no next node)")
            return True

        if self.next_node_channel:
            logger.debug(f"Shard node {self.node_id} already connected to next node.")
            return True

        try:
            # use thunderbolt here if available
            this_properties = await self.discovery.async_get_own_properties()
            thunderbolt_conn = discover_thunderbolt_connection(
                this_properties,
                self.next_node,
            )
            next_ip = (
                thunderbolt_conn.ip_addr
                if thunderbolt_conn
                else self.next_node.local_ip
            )
            address = f"{next_ip}:{self.next_node.shard_port}"
            logger.info(
                f"Shard node {this_properties.instance} connecting to next node {self.next_node.instance} at {address}"
            )

            self.next_node_channel = aio_grpc.insecure_channel(address)
            from ...protos.dnet_ring_pb2_grpc import DnetRingServiceStub

            self.next_node_stub = DnetRingServiceStub(self.next_node_channel)
            return True
        except Exception as e:
            logger.warning(
                f"Shard node {self.node_id} failed to connect to next node {address}: {e}"
            )
            self.next_node_channel = None
            self.next_node_stub = None
            return False

    async def _reconnect_next_node(self) -> bool:
        try:
            if self.next_node_channel:
                await self.next_node_channel.close()
        except Exception:
            pass
        self.next_node_channel = None
        self.next_node_stub = None
        return await self._connect_next_node()

    async def _measure_latency_to_devices(
        self,
        devices: Mapping[str, DnetDeviceProperties],
        thunderbolts: Mapping[str, ThunderboltConnection],
        payload_sizes: list[int],
    ) -> LatencyResults:
        """Measure latency to all devices except self.

        Args:
            devices: Device information mapping
            thunderbolts: Thunderbolt connection information
            payload_sizes: List of payload sizes to test

        Returns:
            Latency measurement results
        """
        latency_results_dict: dict[str, DeviceLatencyResult] = {}

        for instance, device_info in devices.items():
            # Skip measuring latency to ourselves
            if instance == self.discovery.instance_name():
                logger.debug("Skipping latency measurement to self: %s", instance)
                continue

            # Skip measuring latency to API (manager) devices
            if device_info.is_manager:
                logger.debug(
                    "Skipping latency measurement to manager/API: %s", instance
                )
                continue

            try:
                shard_port = device_info.shard_port

                # Check for Thunderbolt connection
                if instance in thunderbolts:
                    tb_data = thunderbolts[instance]
                    instance_tb_ip = tb_data.ip_addr
                    logger.info(
                        "Using Thunderbolt for %s at %s, connected to instance %s",
                        instance,
                        instance_tb_ip,
                        tb_data.instance,
                    )
                else:
                    # No Thunderbolt, use WiFi
                    instance_tb_ip = device_info.local_ip

                if not shard_port or not instance_tb_ip:
                    logger.warning("No shard_port or local_ip for device %s", instance)
                    continue

                # Connect to target shard's gRPC server
                target_address = f"{instance_tb_ip}:{shard_port}"
                channel = aio_grpc.insecure_channel(target_address)
                from ...protos.dnet_ring_pb2_grpc import DnetRingServiceStub

                stub = DnetRingServiceStub(channel)

                # Measure latency for each payload size
                latency_measurements: list[LatencyMeasurement] = []
                for payload_size in payload_sizes:
                    # Create dummy payload
                    dummy_data = b"x" * payload_size

                    start_time = time.perf_counter()
                    timestamp_ms = int(time.time() * 1000)

                    request = dnet_ring_pb2.LatencyMeasureRequest(
                        requester_id=str(self.node_id),
                        payload_size=payload_size,
                        dummy_data=dummy_data,
                        timestamp=timestamp_ms,
                    )

                    response = await stub.MeasureLatency(request)  # type: ignore
                    end_time = time.perf_counter()

                    if response.success:
                        latency_ms = (end_time - start_time) * 1000
                        latency_measurements.append(
                            LatencyMeasurement(
                                payload_size=payload_size,
                                latency_ms=round(latency_ms, 2),
                                success=True,
                                error=None,
                            )
                        )
                    else:
                        latency_measurements.append(
                            LatencyMeasurement(
                                payload_size=payload_size,
                                success=False,
                                error=response.message,
                                latency_ms=0,
                            )
                        )

                # Store results
                result = DeviceLatencyResult(
                    target_node_id=response.node_id if response.success else None,
                    measurements=latency_measurements,
                    success=True,
                    error=None,
                )
                latency_results_dict[instance] = result

                # Close channel
                await channel.close()

            except Exception as e:
                logger.error("Error measuring latency to %s: %s", instance, e)
                result = DeviceLatencyResult(
                    target_node_id=None,
                    success=False,
                    error=str(e),
                    measurements=[],
                )
                latency_results_dict[instance] = result

        return LatencyResults(results=latency_results_dict)
