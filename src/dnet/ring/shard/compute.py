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
from .attrib import RingShardNodeAttributes


class ComputeMixin(RingShardNodeAttributes):
    """Split out the hot-path compute from RingShardNode."""

    def _process_activation(self, activation_msg: ActivationMessage):
        if (
            not self.model
            or not self.model_metadata
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
                did_early_swap = False

                # Determine contiguous local window starting at current_layer
                window_layers: List[int] = []
                _tmp_layer = current_layer
                while processed < self.window_size and (
                    _tmp_layer in self._assigned_set
                ):
                    window_layers.append(_tmp_layer)
                    _tmp_layer += 1
                    processed += 1

                if self._mode == "offload" and window_layers:
                    prep = self._prepared_by_nonce.get(activation_msg.nonce)
                    if prep is not None:
                        layers, fut = prep
                        if layers == window_layers and fut is not None:
                            try:
                                fut.result(timeout=30)
                            except Exception:
                                pass

                # In sliding_fit with a single resident window, proactively evict only the
                # non-needed head from the current resident set before loading new weights.
                # This prevents LRU from evicting the useful tail during materialization.
                if self._mode == "sliding_fit" and int(self._resident_windows) <= 1 and window_layers:
                    try:
                        # Current requested window
                        curr = list(window_layers)
                        # Budget equals window_size for single resident window
                        try:
                            budget = max(1, int(self.window_size) or 1)
                        except Exception:
                            budget = max(1, int(self.window_size))
                        # Derive actual resident layers from cache ordered oldest->newest
                        resident = []
                        try:
                            resident = self.weight_cache.get_resident_layers()  # type: ignore[union-attr]
                        except Exception:
                            resident = []
                        # Consider only layers not part of the current window
                        prev_only = [lid for lid in resident if lid not in curr]
                        keep_quota = max(0, budget - len(curr))
                        keep_tail = prev_only[-keep_quota:] if keep_quota > 0 else []
                        evict_head = [lid for lid in prev_only if lid not in set(keep_tail)]
                        if evict_head:
                            try:
                                evicted_cnt = self.weight_cache.evict_layers(evict_head)  # type: ignore[union-attr]
                            except Exception:
                                evicted_cnt = 0
                            try:
                                if hasattr(self.model, "unload_layers"):
                                    self.model.unload_layers(evict_head)  # type: ignore[attr-defined]
                                    for lid in evict_head:
                                        self._bound_versions.pop(lid, None)
                            except Exception:
                                pass
                            did_early_swap = True
                            if self._profile:
                                try:
                                    logger.info(
                                        "[PROFILE][DELTA-SWAP][EARLY] node=%s nonce=%s evict_head=%s keep_tail=%s add=%s evicted=%s",
                                        self.node_id,
                                        activation_msg.nonce,
                                        evict_head,
                                        keep_tail,
                                        window_layers,
                                        evicted_cnt,
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # Ensure weights for the window are resident and bind only if arrays changed
                # if model fits and we've already bound these layers, skip the scan entirely.
                fast_fit = (self._mode == "fit" and len(self._assigned_sorted) <= self.window_size)
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
                t_comp = time.perf_counter()
                try:
                    self._compute_busy.set()
                except Exception:
                    pass
                for i, lyr in enumerate(window_layers):
                    with self._mlx_lock:
                        x = self.model.apply_single_layer(lyr, x, cache=kv)
                        try:
                            if str(x.dtype) != str(self._wire_mx_dtype):
                                x = x.astype(self._wire_mx_dtype)
                        except Exception:
                            pass

                last_layer = (
                    window_layers[-1] if window_layers else activation_msg.layer_id
                )
                try:
                    mx.eval(x)
                except Exception:
                    pass
                if self._profile:
                    t_comp_done = time.perf_counter()
                    logger.info(
                        "[PROFILE][WINDOW] node=%s nonce=%s layers=%s compute_ms=%.3f",
                        self.node_id,
                        activation_msg.nonce,
                        window_layers,
                        (t_comp_done - t_comp) * 1000.0,
                    )

                for lid in window_layers:
                    self.weight_cache.decrease_reference(lid)

                try:
                    # Sliding-fit delta swap: maintain a single resident set by evicting
                    # only what's needed to fit the next window into the budget. Prefer
                    # keeping the tail of the previous window so we don't thrash weights
                    # that are likely to be reused.
                    if self._mode == "sliding_fit":
                        if int(self._resident_windows) <= 1:
                            if did_early_swap:
                                # Early delta-swap already trimmed resident set for this window
                                pass
                            elif not self._recent_windows:
                                # First window in token: seed resident set
                                self._recent_windows.append(list(window_layers))
                            else:
                                prev = self._recent_windows.pop(0)
                                # Compute minimal eviction to fit the new window into budget.
                                # Budget equals window_size when resident_windows == 1.
                                try:
                                    budget = max(1, int(self.window_size) or 1)
                                except Exception:
                                    budget = max(1, int(self.window_size))

                                curr = list(window_layers)
                                prev_only = [x for x in prev if x not in curr]
                                keep_quota = max(0, budget - len(curr))
                                keep_tail = prev_only[-keep_quota:] if keep_quota > 0 else []
                                evict_head = [x for x in prev_only if x not in set(keep_tail)]
                                try:
                                    evicted_cnt = self.weight_cache.evict_layers(evict_head)
                                except Exception:
                                    evicted_cnt = 0
                                try:
                                    if hasattr(self.model, "unload_layers"):
                                        self.model.unload_layers(evict_head)  # type: ignore[attr-defined]
                                        for lid in evict_head:
                                            self._bound_versions.pop(lid, None)
                                except Exception:
                                    pass
                                combined = list(keep_tail) + curr
                                self._recent_windows.append(combined)
                                if self._profile:
                                    try:
                                        logger.info(
                                            "[PROFILE][DELTA-SWAP] node=%s nonce=%s evict_head=%s keep_tail=%s add=%s evicted=%s",
                                            self.node_id,
                                            activation_msg.nonce,
                                            evict_head,
                                            keep_tail,
                                            window_layers,
                                            evicted_cnt,
                                        )
                                    except Exception:
                                        pass
                        else:
                            # resident_windows>1 not expected in sliding_fit; fall back to seeding
                            self._recent_windows.append(list(window_layers))
                    else:
                        # Original eviction policy for other modes
                        self._recent_windows.append(list(window_layers))
                        if int(self._resident_windows) <= 1:
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
                        else:
                            if not self._defer_unload:
                                while len(self._recent_windows) > max(1, int(self._resident_windows)):
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

                nxt = last_layer + 1
                if nxt >= self.model_metadata.num_layers:  # End of model
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
                # Enqueue to appropriate asyncio TX queue from compute thread
                try:
                    if self._loop is not None:
                        target_q = (
                            self.activation_token_queue
                            if output_msg.is_final
                            else self.activation_computed_queue
                        )
                        fut = asyncio.run_coroutine_threadsafe(
                            target_q.put(output_msg), self._loop
                        )
                        fut.result()
                    else:
                        raise RuntimeError("Event loop not available for TX queue")
                except Exception as e:
                    logger.error(
                        "Failed to queue computed activation for sending: %s", e
                    )
                
                # Clean up input resources
                self.input_pool.release(activation_msg.pool_id)
                

                # Optional unload/evict after stage
                if self._mode != "sliding_fit":
                    if self._defer_unload:
                        try:
                            while len(self._recent_windows) > max(1, int(self._resident_windows)):
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

                    if self._resident_windows <= 1:
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
                return
        except Exception as e:
            logger.exception("Error processing activation: %s", e)
