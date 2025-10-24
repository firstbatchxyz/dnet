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

    def _delta_swap_eviction(
        self,
        window_layers: List[int],
        resident: List[int],
        activation_msg: ActivationMessage,
        early: bool = False,
    ) -> int:
        budget = max(1, int(self.window_size or 1))
        curr_set = set(window_layers)
        prev_only = [lid for lid in resident if lid not in curr_set]
        keep_quota = max(0, budget - len(window_layers))
        idx = max(0, len(prev_only) - keep_quota)
        keep_tail = prev_only[idx:]
        evict_head = prev_only[:idx]
        if not evict_head:
            return 0
        evicted: List[int] = []
        for lid in evict_head:
            try:
                if self.weight_cache.evict_layer(lid):
                    evicted.append(lid)
            except Exception:
                continue
        if evicted:
            try:
                if hasattr(self.model, "unload_layers"):
                    self.model.unload_layers(evicted)
                for lid in evicted:
                    self._bound_versions.pop(lid, None)
            except Exception:
                pass
            if self._profile:
                try:
                    tag = "DELTA-SWAP-EARLY" if early else "DELTA-SWAP"
                    logger.info(
                        f"[PROFILE][{tag}] node=%s nonce=%s evict_head=%s keep_tail=%s add=%s evicted=%s",
                        self.node_id,
                        activation_msg.nonce,
                        evict_head,
                        keep_tail,
                        window_layers,
                        len(evicted),
                    )
                except Exception:
                    pass
        return len(evicted)

    def _process_activation(self, activation_msg: ActivationMessage):
        if (not self.model
            or not self.model_metadata
            or not self.weight_cache
            or not self.input_pool
            or not self.output_pool
        ):
            logger.error("Node %s: Cannot process activation - model not loaded", self.node_id)
            return

        try:
            # per-nonce kvcache for concurrent requests
            with self.tracer.frame("compute.thread", "kvcache.init"):
                kv = self._get_or_make_kv(activation_msg.nonce)

            # Get input activation from pool
            with self.tracer.frame("compute.thread", "activations.load"):
                input_buffer = self.input_pool.get_buffer(activation_msg.pool_id)
                if input_buffer is None:
                    logger.error("Failed to get input buffer %s", activation_msg.pool_id)
                    return

            # Prepare input activation
            with self.tracer.frame("compute.thread", "activations.process") as f:
                f.set("nonce", activation_msg.nonce)
                if activation_msg.dtype == "tokens": # embed locally on start shard
                    f.event("embed_tokens")
                    numel = int(np.prod(activation_msg.shape))
                    tok_view = input_buffer[:numel].reshape(activation_msg.shape)
                    toks = mx.array(np.array(tok_view, dtype=np.int32), dtype=mx.int32)
                    x = self.model.embed(toks[None])

                    # NOTE: Used to track start of request in perf stats 
                    self.tracer.mark("embedding", {
                      "nonce": activation_msg.nonce,
                      "prompt_tokens": toks.size,
                    }) 

                    if x.dtype != self._wire_mx_dtype:
                        x = x.astype(self._wire_mx_dtype)

                else: # Prepare input activation using MLX view operations only
                    f.set("activation_dtype", activation_msg.dtype)
                    numel = int(np.prod(activation_msg.shape))
                    x = input_buffer[:numel].reshape(activation_msg.shape)

                    try: # Ensure expected dtype 
                        if str(x.dtype) != activation_msg.dtype:
                            x = x.astype(mlx_dtype_map[activation_msg.dtype])
                    except Exception:
                        logger.warning(f"Unable to update activation dtype")

            # Compute windows until boundary (stay local as long as possible)
            current_layer = activation_msg.layer_id + 1
            last_layer = current_layer - 1
            while True:
                processed = 0
                did_early_swap = False

                with self.tracer.frame("compute.thread", "weights.prepare") as f:

                    # Determine contiguous local window starting at current_layer
                    window_layers: List[int] = []
                    _tmp_layer = current_layer
                    while processed < self.window_size and (_tmp_layer in self._assigned_set):
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
                    if (
                        self._mode == "sliding_fit"
                        and int(self._resident_windows) <= 1
                        and window_layers
                    ):
                        try:
                            resident = []
                            try:
                                resident = self.weight_cache.get_resident_layers()  # type: ignore[union-attr]
                            except Exception:
                                resident = []
                            evicted_cnt = self._delta_swap_eviction(
                                window_layers, resident, activation_msg, early=True
                            )
                            if evicted_cnt > 0:
                                did_early_swap = True
                        except Exception:
                            pass

                    # Ensure weights for the window are resident and bind only if arrays changed
                    # if model fits and we've already bound these layers, skip the scan entirely.
                    fast_fit = (
                        self._mode == "fit"
                        and len(self._assigned_sorted) <= self.window_size
                    )
                    skip_scan = fast_fit and all(
                        (wl in self._bound_versions) for wl in window_layers
                    )
                    to_bind: Dict[str, mx.array] = {}
                    if not skip_scan:
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

                        # Opportunistically schedule prefetch for the next window to overlap with compute
                        try:
                            next_win_pre = self._next_local_layers(
                                (window_layers[-1] if window_layers else (activation_msg.layer_id)),
                                self.window_size,
                            )
                            for nl in next_win_pre:
                                self._prefetch_to_ram(nl)
                                self._enqueue_weight_prefetch(nl)
                        except Exception:
                            pass

                # Execute the window
                with self.tracer.frame("compute.thread", "execute"):
                    self._beyond_cursor = window_layers[-1] if window_layers else (activation_msg.layer_id)

                    try: # Prevent prefetch touching during encode/compute to minimize UMA pressure
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
                """

                for lid in window_layers:
                    #self.weight_cache.decrease_reference(lid)
                    pass

                with self.tracer.frame("compute.thread", "execute.evict_and_unload"):
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
                                    self._delta_swap_eviction(window_layers, prev, activation_msg, early=False)
                                    budget = max(1, int(self.window_size or 1))
                                    curr = list(window_layers)
                                    prev_only = [x for x in prev if x not in curr]
                                    keep_quota = max(0, budget - len(curr))
                                    keep_tail = prev_only[-keep_quota:] if keep_quota > 0 else []
                                    combined = list(keep_tail) + curr
                                    self._recent_windows.append(combined)
                            else:
                                prev = self._recent_windows.pop(0)
                                self._delta_swap_eviction(
                                    window_layers, prev, activation_msg, early=False
                                )
                                budget = max(1, int(self.window_size or 1))
                                curr = list(window_layers)
                                prev_only = [x for x in prev if x not in curr]
                                keep_quota = max(0, budget - len(curr))
                                keep_tail = (
                                    prev_only[-keep_quota:] if keep_quota > 0 else []
                                )
                                combined = list(keep_tail) + curr
                                self._recent_windows.append(combined)
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
                        else:
                            if not self._defer_unload:
                                while len(self._recent_windows) > max(
                                    1, int(self._resident_windows)
                                ):
                                    old = self._recent_windows.pop(0)
                                    try:
                                        evicted_cnt = self.weight_cache.evict_layers(
                                            old
                                        )
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

                # If next layer is still local, continue without staging/tx
                nxt = last_layer + 1
                if nxt in self._assigned_set:
                    current_layer = nxt
                    continue

                # Boundary reached — directly pass tensor to TX to avoid extra copy/sync
                with self.tracer.frame("compute.thread", "execute.enqueue_prefetch"):
                    x_cast = x if x.dtype == self._wire_mx_dtype else x.astype(self._wire_mx_dtype)
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

                # Create and enqueue output message: either forward activations or finalize on end role
                with self.tracer.frame("compute.thread", "grpc.send"):
                    nxt = last_layer + 1
                    if nxt >= self.model_metadata.num_layers:  # End of model
                        try:
                            with self._mlx_lock:
                                y = self.model.normalize(x_cast)
                                y = self.model.lm_project(y)
                                #self.tracer.mark("lm_head", {"nonce": actication_msg.nonce}) # NOTE: canonical stats end 

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
                with self.tracer.frame("compute.thread", "cleanup"):
                    if self._mode != "sliding_fit":
                        if self._defer_unload:
                            try:
                                while len(self._recent_windows) > max(
                                    1, int(self._resident_windows)
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
