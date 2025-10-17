from __future__ import annotations

import time
import asyncio
from typing import Dict, List, cast

import mlx.core as mx
import numpy as np

from ...utils.logger import logger
from ...utils.serialization import mlx_dtype_map
from ...utils.time import utc_epoch_now
from ..data_types import ActivationMessage


class ComputeMixin:
    """Split out the hot-path compute from RingShardNode."""

    def _process_activation(self, activation_msg: ActivationMessage):
        if (
            not self._check_model_loaded()
            or not self.weight_cache
            or not self.input_pool
            or not self.output_pool
        ):
            logger.error(
                "Node %s: Cannot process activation - model not loaded", self.node_id
            )
            return

        try:
            # per-nonce kvcache for concurrent requests
            kv = self._get_or_make_kv(activation_msg.nonce)

            # Get input activation from pool
            input_buffer = self.input_pool.get_buffer(activation_msg.pool_id)
            if input_buffer is None:
                logger.error("Failed to get input buffer %s", activation_msg.pool_id)
                return

            # Prepare input activation
            if activation_msg.dtype == "tokens":
                # tokens were staged as int32 in the pool; embed locally on start shard
                input_size = int(np.prod(activation_msg.shape))
                tok_view = input_buffer[:input_size].reshape(activation_msg.shape)
                # Convert robustly to MLX int32 and embed (batch=1)
                toks = mx.array(np.array(tok_view, dtype=np.int32), dtype=mx.int32)
                x = self.model.embed(toks[None])
                if x.dtype != self._wire_mx_dtype:
                    x = x.astype(self._wire_mx_dtype)
            else:
                # Prepare input activation using MLX view operations only
                input_size = int(np.prod(activation_msg.shape))
                x = input_buffer[:input_size].reshape(activation_msg.shape)
                # Ensure expected dtype without re-materializing when not needed
                try:
                    if str(x.dtype) != activation_msg.dtype:
                        x = x.astype(mlx_dtype_map[activation_msg.dtype])
                except Exception:
                    pass

            # Compute windows until boundary (stay local as long as possible)
            current_layer = activation_msg.layer_id + 1
            last_layer = current_layer - 1
            while True:
                start_time = time.perf_counter()
                processed = 0

                # Determine contiguous local window starting at current_layer
                window_layers: List[int] = []
                _tmp_layer = current_layer
                while processed < self.window_size and (
                    _tmp_layer in self._assigned_set
                ):
                    window_layers.append(_tmp_layer)
                    _tmp_layer += 1
                    processed += 1

                # Ensure weights for the window are resident and bind only if arrays changed
                # if model fits and we've already bound these layers, skip the scan entirely.
                fast_fit = (
                    getattr(self, "_mode", "") == "fit"
                    and len(self._assigned_sorted) <= self.window_size
                )
                skip_scan = fast_fit and all(
                    (wl in self._bound_versions) for wl in window_layers
                )
                to_bind: Dict[str, mx.array] = {}
                if not skip_scan:
                    t_w_ready = time.perf_counter()
                    for wl in window_layers:
                        weights = self.weight_cache.get_weight(wl)
                        if weights is None:
                            logger.error("Failed to load weights for layer %s", wl)
                            self.input_pool.release(activation_msg.pool_id)
                            return
                        try:
                            # Use identity of first array as a cheap version/fingerprint
                            first_arr = next(iter(weights.values()))
                            version = id(first_arr)
                        except StopIteration:
                            version = -1
                        if self._bound_versions.get(wl) != version:
                            for k, v in weights.items():
                                to_bind[k] = v
                            self._bound_versions[wl] = version
                    if self._profile:
                        t_w_ms = (time.perf_counter() - t_w_ready) * 1000.0
                        # Only log when non-trivial or binding happened to reduce overhead/noise
                        if to_bind or t_w_ms > 0.5:
                            logger.info(
                                "[PROFILE][WAIT-WEIGHTS] node=%s nonce=%s layers=%s ms=%.3f",
                                self.node_id,
                                activation_msg.nonce,
                                window_layers,
                                t_w_ms,
                            )

                bind_ms = 0.0
                if to_bind:
                    # Block prefetch-touch during binding and serialize MLX ops
                    try:
                        self._compute_busy.set()
                    except Exception:
                        pass
                    t_bind = time.perf_counter()
                    with self._mlx_lock:
                        self.model.load_weights(list(to_bind.items()), strict=False)
                    bind_ms = (time.perf_counter() - t_bind) * 1000.0
                    if self._profile:
                        logger.info(
                            "[PROFILE][BIND] node=%s nonce=%s layers=%s tensors=%s bind_ms=%.3f",
                            self.node_id,
                            activation_msg.nonce,
                            window_layers,
                            len(to_bind),
                            bind_ms,
                        )

                # Opportunistically schedule prefetch for the next window to overlap with compute
                try:
                    next_win_pre = self._next_local_layers(
                        (
                            window_layers[-1]
                            if window_layers
                            else (activation_msg.layer_id)
                        ),
                        self.window_size,
                    )
                    for nl in next_win_pre:
                        self._prefetch_to_ram(nl)
                        self._enqueue_weight_prefetch(nl)
                    if self._profile:
                        logger.info(
                            "[PROFILE][PREFETCH-SCHED] node=%s nonce=%s next_window(pre)=%s",
                            self.node_id,
                            activation_msg.nonce,
                            next_win_pre,
                        )
                except Exception:
                    pass

                # Execute the window
                self._beyond_cursor = (
                    window_layers[-1] if window_layers else (activation_msg.layer_id)
                )
                t_comp = time.perf_counter()
                # Prevent prefetch touching during encode/compute to minimize UMA pressure
                try:
                    self._compute_busy.set()
                except Exception:
                    pass
                layer_times_ms: list[tuple[int, float]] = []
                for i, lyr in enumerate(window_layers):
                    t_l0 = (
                        time.perf_counter() if getattr(self, "_profile", False) else 0.0
                    )
                    with self._mlx_lock:
                        x = self.model.apply_single_layer(lyr, x, cache=kv)

                    # Optional per-n-layer sync for profiling, gated by settings
                    if getattr(self, "_profile", False) and getattr(
                        self, "_sync_per_layer", False
                    ):
                        do_sync = True
                        try:
                            n = int(getattr(self, "_sync_every_n", 0) or 0)
                        except Exception:
                            n = 0
                        if n > 0 and (i % n) != 0:
                            do_sync = False
                        if do_sync:
                            try:
                                with self._mlx_lock:
                                    _s = mx.sum(x)
                                    mx.eval(_s)
                            except Exception:
                                pass
                            dt = (time.perf_counter() - t_l0) * 1000.0
                            layer_times_ms.append((lyr, dt))
                            self._prof.info(
                                "[PROFILE][LAYER] node=%s nonce=%s layer=%s ms=%.3f",
                                self.node_id,
                                activation_msg.nonce,
                                lyr,
                                dt,
                            )
                t_comp_done = time.perf_counter()
                last_layer = (
                    window_layers[-1] if window_layers else activation_msg.layer_id
                )
                # Ensure compute is complete before any cache/eviction ops
                # try:
                #    mx.eval(x)
                # except Exception:
                #    pass
                if self._profile:
                    try:
                        avg = (
                            sum(t for _, t in layer_times_ms) / len(layer_times_ms)
                            if layer_times_ms
                            else 0.0
                        )
                        self._prof.info(
                            "[PROFILE][LAYER-AVG] node=%s nonce=%s window=%s avg_ms=%.3f per_layer_ms=%.3f",
                            self.node_id,
                            activation_msg.nonce,
                            window_layers,
                            avg,
                            (avg / len(layer_times_ms)) if layer_times_ms else 0.0,
                        )
                    except Exception:
                        pass
                    self._prof.info(
                        "[PROFILE][EXEC-PATH] node=%s nonce=%s compiled=%s window=%s",
                        self.node_id,
                        activation_msg.nonce,
                        0,
                        window_layers,
                    )
                    # Optional activation stats at window boundary for debugging
                    try:
                        if getattr(self, "_x_stats", False):
                            m = mx.mean(x)
                            s = mx.std(x)
                            mn = mx.min(x)
                            mxv = mx.max(x)
                            mx.eval(m, s, mn, mxv)
                            logger.info(
                                "[PROFILE][X-STATS] node=%s nonce=%s window_tail=%s mean=%s std=%s min=%s max=%s",
                                self.node_id,
                                activation_msg.nonce,
                                last_layer,
                                float(m.item()),
                                float(s.item()),
                                float(mn.item()),
                                float(mxv.item()),
                            )
                    except Exception:
                        pass
                    logger.info(
                        "[PROFILE][WINDOW] node=%s nonce=%s layers=%s compute_ms=%.3f",
                        self.node_id,
                        activation_msg.nonce,
                        window_layers,
                        (t_comp_done - t_comp) * 1000.0,
                    )

                # Decrease references post-compute. Defer any eviction/unload of the
                # just-computed window until after we synchronize (stage barrier)
                # to avoid forcing MLX to re-materialize weights during eval.
                for lid in window_layers:
                    self.weight_cache.decrease_reference(lid)

                try:
                    self._recent_windows.append(list(window_layers))
                    if not self._defer_unload:
                        while len(self._recent_windows) > max(
                            1, int(getattr(self, "_resident_windows", 2))
                        ):
                            old = self._recent_windows.pop(0)
                            # Proactively evict; shrink params for old window
                            try:
                                evicted_cnt = self.weight_cache.evict_layers(old)
                            except Exception:
                                evicted_cnt = 0
                            try:
                                if hasattr(self.model, "unload_layers"):
                                    self.model.unload_layers(old)  # type: ignore[attr-defined]
                                    for lid in old:
                                        self._bound_versions.pop(lid, None)
                            except Exception:
                                pass
                            if self._profile:
                                try:
                                    logger.info(
                                        "[PROFILE][UNLOAD-WINDOW] node=%s nonce=%s old_layers=%s evicted=%s keep_windows=%s",
                                        self.node_id,
                                        activation_msg.nonce,
                                        old,
                                        evicted_cnt,
                                        self._resident_windows,
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass

                computation_time = time.perf_counter() - start_time
                self._prof.info(
                    "[PROFILE][COMPUTE] node=%s nonce=%s window_layers=%s total_ms=%.3f",
                    self.node_id,
                    activation_msg.nonce,
                    window_layers,
                    computation_time * 1000.0,
                )
                self._prof.info(
                    "Completed layers up to %s in %.3fs, nonce: %s, result: %s %s",
                    last_layer,
                    computation_time,
                    activation_msg.nonce,
                    x.shape,
                    x.dtype,
                )

                # If next layer is still local, continue without staging/tx
                nxt = last_layer + 1
                if nxt in self._assigned_set:
                    current_layer = nxt
                    continue

                # Boundary reached — directly pass tensor to TX to avoid extra copy/sync
                t_stage = time.perf_counter()
                x_cast = (
                    x
                    if x.dtype == self._wire_mx_dtype
                    else x.astype(self._wire_mx_dtype)
                )
                try:
                    self._compute_busy.clear()
                except Exception:
                    pass
                try:
                    for lid in list(self._prefetch_pending):
                        self._prefetch_pending.discard(lid)
                        self._enqueue_weight_prefetch(lid)
                except Exception:
                    pass
                if self._profile:
                    try:
                        logger.info(
                            "[PROFILE][STAGE-DIRECT] node=%s nonce=%s layer_tail=%s stage_ms=%.3f shape=%s dtype=%s",
                            self.node_id,
                            activation_msg.nonce,
                            last_layer,
                            (time.perf_counter() - t_stage) * 1000.0,
                            tuple(x_cast.shape),
                            str(self._wire_mx_dtype),
                        )
                    except Exception:
                        pass

                # Create and enqueue output message: either forward activations or finalize on end role
                nxt = last_layer + 1
                if (
                    nxt >= self.model_metadata.num_layers
                    and getattr(self, "role", "inter") == "end"
                ):
                    # End-shard head+sampling inline; return only token to API
                    try:
                        with self._mlx_lock:
                            y = self.model.normalize(x_cast)
                            y = self.model.lm_project(y)
                        # Greedy sample last position
                        if y.ndim == 3:
                            logits_2d = y[:, -1, :]
                        elif y.ndim == 2:
                            logits_2d = y[-1:, :]
                        else:
                            logits_2d = y.reshape(1, -1)
                        tok = mx.argmax(logits_2d, axis=-1)
                        token_id = int(tok.item())
                    except Exception as e:
                        logger.error("End-shard sampling failed: %s", e)
                        return
                    output_msg = ActivationMessage(
                        nonce=activation_msg.nonce,
                        layer_id=last_layer,
                        pool_id=-1,
                        shape=cast(tuple[int, ...], x.shape),
                        batch_size=activation_msg.batch_size,
                        timestamp=utc_epoch_now(),
                        node_origin=f"node_{self.node_id}",
                        dtype=str(self._wire_mx_dtype),
                        callback_url=activation_msg.callback_url,
                        is_final=True,
                        token_id=token_id,
                    )
                else:
                    output_msg = ActivationMessage(
                        nonce=activation_msg.nonce,
                        layer_id=last_layer,
                        pool_id=-1,
                        shape=cast(tuple[int, ...], x.shape),
                        batch_size=activation_msg.batch_size,
                        timestamp=utc_epoch_now(),
                        node_origin=f"node_{self.node_id}",
                        dtype=str(self._wire_mx_dtype),
                        callback_url=activation_msg.callback_url,
                        tensor=x_cast,
                    )
                try:
                    output_msg.tx_enq_perf_t = time.perf_counter()
                except Exception:
                    output_msg.tx_enq_perf_t = 0.0
                # Enqueue to asyncio TX queue from compute thread
                try:
                    loop = getattr(self, "_loop", None)
                    if loop is not None:
                        fut = asyncio.run_coroutine_threadsafe(
                            self.activation_computed_queue.put(output_msg), loop
                        )
                        fut.result(timeout=10)
                    else:
                        # Fallback: try immediate put_nowait via a temporary loop context
                        # (should not happen in practice)
                        raise RuntimeError("Event loop not available for TX queue")
                except Exception as e:
                    logger.error(
                        "Failed to queue computed activation for sending: %s", e
                    )
                    # nothing to release when using direct tensor path

                # Clean up input resources
                self.input_pool.release(activation_msg.pool_id)
                # After queuing TX, schedule prefetch and eviction in the background
                # to avoid stalling the handoff to the next shard.
                try:
                    self._prefetch_pause.set()
                except Exception:
                    pass
                next_window = self._next_local_layers(last_layer, self.window_size)
                for nl in next_window:
                    self._prefetch_to_ram(nl)
                    self._enqueue_weight_prefetch(nl)
                if self._profile:
                    logger.info(
                        "[PROFILE][PREFETCH-SCHED] node=%s nonce=%s next_window=%s",
                        self.node_id,
                        activation_msg.nonce,
                        next_window,
                    )

                # Optional unload/evict after stage
                if getattr(self, "_defer_unload", False):
                    try:
                        while len(self._recent_windows) > max(
                            1, int(getattr(self, "_resident_windows", 2))
                        ):
                            old = self._recent_windows.pop(0)
                            try:
                                evicted_cnt = self.weight_cache.evict_layers(old)
                            except Exception:
                                evicted_cnt = 0
                            try:
                                if hasattr(self.model, "unload_layers"):
                                    self.model.unload_layers(old)  # type: ignore[attr-defined]
                                    for lid in old:
                                        self._bound_versions.pop(lid, None)
                            except Exception:
                                pass
                            if self._profile:
                                logger.info(
                                    "[PROFILE][UNLOAD-WINDOW] node=%s nonce=%s old_layers=%s evicted=%s keep_windows=%s (post-stage)",
                                    self.node_id,
                                    activation_msg.nonce,
                                    old,
                                    evicted_cnt,
                                    self._resident_windows,
                                )
                    except Exception:
                        pass

                if getattr(self, "_resident_windows", 2) <= 1:
                    try:
                        evicted = self.weight_cache.evict_layers(window_layers)
                        if hasattr(self.model, "unload_layers"):
                            self.model.unload_layers(window_layers)  # type: ignore[attr-defined]
                        if self._profile:
                            logger.info(
                                "[PROFILE][EVICT] node=%s nonce=%s layers=%s evicted=%s",
                                self.node_id,
                                activation_msg.nonce,
                                window_layers,
                                evicted,
                            )
                    except Exception:
                        pass

                for nl in next_window:
                    self._enqueue_weight_prefetch(nl)
                # Finished all local work for this activation
                return

        except Exception as e:
            logger.exception("Error processing activation: %s", e)
