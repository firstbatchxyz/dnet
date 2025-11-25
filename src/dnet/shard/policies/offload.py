from typing import cast
from dnet.core.memory.weight_cache import WeightCache
from ..models import ShardLoadModelRequest
from dnet.core.types.messages import ActivationMessage
from dnet.core.decoding.sampler import Sampler
from dnet.core.decoding.config import DecodingConfig
from dnet.utils.logger import logger
import mlx.core as mx
import numpy as np
from dnet.utils.serialization import mlx_dtype_map
from dnet.utils.time import utc_epoch_now
from .base import register_policy, ComputePolicy
from dnet.utils.repack import ensure_repacked_for_layers
from dnet.utils.model import get_model_metadata
import time
import asyncio


@register_policy("offload")
@register_policy("sliding_fit")
class OffloadPolicy(ComputePolicy):
    """
    Policy for offloading weights or sliding window fit.
    Handles 'offload' and 'sliding_fit' modes.
    """

    def configure_policy_for_model(self, req: ShardLoadModelRequest) -> None:
        local_count = max(1, len(self.runtime.assigned_layers))
        requested_w = max(1, int(req.window_size))
        n_residency = int(max(1, int(req.residency_size)))

        if n_residency < requested_w:
            self._mode = "sliding_fit"
            self.window_size = max(1, min(n_residency, local_count))
            self._resident_windows = (
                int(self._resident_windows) if self._resident_windows else 1
            )
        else:
            self._mode = "offload"
            self.window_size = max(1, min(requested_w, local_count))
            self._resident_windows = (
                int(self._resident_windows) if self._resident_windows else 1
            )

        # Repack for offload/sliding_fit
        try:
            t0 = time.perf_counter()
            logger.info("Repacking model weights (this may take a while)...")
            repacked_dir, did_repack = ensure_repacked_for_layers(
                self.runtime.model_path, self.runtime._assigned_sorted
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.runtime.model_path = str(repacked_dir)
            self.runtime.model_metadata = get_model_metadata(self.runtime.model_path)

            self.runtime.compute_config.mxload_fastpath = True
            self.runtime.compute_config.prefetch_mode = "off"

            logger.info(
                "[REPACK] shard=%s dst=%s layers=%s repacked=%s ms=%.1f",
                self.runtime.shard_id,
                self.runtime.model_path,
                len(self.runtime._assigned_sorted),
                int(did_repack),
                dt_ms,
            )
        except Exception as e:
            logger.warning(
                "Runtime %s: repack failed or skipped: %s", self.runtime.shard_id, e
            )

        # For offload/sliding_fit, we typically disable auto-prefetch in WeightCache
        # and handle it explicitly or rely on blocking loads.
        # The old shard set prefetch_mode="off" for these modes.
        prefetch_mode = "off"

        # Enable fastpath for these modes as per old shard
        use_fastpath = True

        self.weight_cache = WeightCache(
            self.runtime.assigned_layers,
            self.runtime.model_metadata,
            window_size=self.window_size,
            prefetch_threads=self.runtime.prefetch_threads,
            resident_windows=self._resident_windows,
            use_mxload_fastpath=use_fastpath,
            prefetch_mode=prefetch_mode,
        )

        logger.info(
            "OffloadPolicy configured: mode=%s window=%d resident=%d",
            self._mode,
            self.window_size,
            self._resident_windows,
        )

    def _prepare_window_blocking(self, window_layers: list[int]) -> None:
        """Synchronously materialize the given window's weights to device memory.

        This runs in the thread pool to avoid blocking the event loop.
        """
        try:
            if not self.weight_cache:
                return
            for lid in window_layers:
                _ = self.weight_cache.get_weight(lid, inc_ref=False)
        finally:
            pass

    def process(self, msg: ActivationMessage) -> None:
        if (
            not self.runtime.model
            or not self.runtime.model_metadata
            or not self.weight_cache
            or not self.runtime.input_pool
            or not self.runtime.output_pool
        ):
            logger.error(
                "Runtime %s: cannot process activation - model not loaded",
                self.runtime.shard_id,
            )
            return

        try:
            with self.runtime._model_lock:
                if (
                    not self.runtime.model
                    or not self.runtime.model_metadata
                    or not self.weight_cache
                    or not self.runtime.input_pool
                    or not self.runtime.output_pool
                ):
                    logger.error(
                        "Runtime %s: cannot process activation - model not loaded",
                        self.runtime.shard_id,
                    )
                    return

                # 1) per nonce KV
                kv = self.runtime.get_or_make_kv(msg.nonce)

                # 2) get input tensor from pool
                input_buffer = self.runtime.input_pool.get_buffer(msg.pool_id)
                if input_buffer is None:
                    logger.error("Failed to get input buffer %s", msg.pool_id)
                    return

                # 3) prepare x
                input_size = int(np.prod(msg.shape))
                reshaped_data = input_buffer[:input_size].reshape(msg.shape)

                if msg.dtype == "tokens":
                    # Token path: convert to int32, embed, and ensure correct dtype
                    toks = mx.array(
                        np.array(reshaped_data, dtype=np.int32), dtype=mx.int32
                    )
                    x = self.runtime.model.embed(toks[None])
                    target_dtype = self.runtime._wire_mx_dtype
                else:
                    # Non-token path: use data as-is, convert dtype if needed
                    x = reshaped_data
                    target_dtype = mlx_dtype_map[msg.dtype]

                if target_dtype and x.dtype != target_dtype:
                    x = x.astype(target_dtype)

                current_layer = msg.layer_id + 1

                while True:
                    did_early_swap = False

                    # build contiguous window inside our shard
                    window_layers: list[int] = []
                    for i in range(self.window_size):
                        layer = current_layer + i
                        if layer not in self.runtime._assigned_set:
                            break
                        window_layers.append(layer)

                    if not window_layers:
                        break

                    # Wait for prefetch if available (offload mode)
                    if self._mode == "offload" and window_layers:
                        prep = self._prepared_by_nonce.get(msg.nonce)
                        if prep is not None:
                            layers, fut = prep
                            if layers == window_layers and fut is not None:
                                try:
                                    fut.result(timeout=30)
                                except Exception:
                                    pass

                    # Early eviction for sliding_fit
                    if (
                        self._mode == "sliding_fit"
                        and int(self._resident_windows) <= 1
                        and window_layers
                    ):
                        try:
                            try:
                                resident = self.weight_cache.get_resident_layers()
                            except Exception:
                                resident = []
                            evicted_cnt = self._delta_swap_eviction(
                                window_layers, resident
                            )
                            if evicted_cnt > 0:
                                did_early_swap = True
                        except Exception:
                            pass

                    # Bind weights
                    to_bind = self._bind_layer_weights(window_layers, msg)
                    if to_bind is None:
                        return
                    if to_bind:
                        self.runtime._compute_busy.set()
                        try:
                            with self.runtime._mlx_lock:
                                self.runtime.model.load_weights(
                                    list(to_bind.items()), strict=False
                                )
                        finally:
                            self.runtime._compute_busy.clear()

                    # compute window
                    try:
                        self.runtime._compute_busy.set()
                        for lyr in window_layers:
                            with self.runtime._mlx_lock:
                                x = self.runtime.model.apply_single_layer(
                                    lyr, x, cache=kv
                                )
                                try:
                                    if str(x.dtype) != str(self.runtime._wire_mx_dtype):
                                        x = x.astype(self.runtime._wire_mx_dtype)
                                except Exception:
                                    pass
                    finally:
                        self.runtime._compute_busy.clear()

                    last_layer = window_layers[-1]
                    try:
                        mx.eval(x)
                    except Exception:
                        pass

                    for lid in window_layers:
                        self.weight_cache.decrease_reference(lid)

                    # Eviction logic
                    try:
                        if self._mode == "sliding_fit":
                            if int(self._resident_windows) <= 1:
                                if did_early_swap:
                                    pass
                                elif not self._recent_windows:
                                    self._recent_windows.append(list(window_layers))
                                else:
                                    prev = self._recent_windows.pop(0)
                                    self._delta_swap_eviction(window_layers, prev)

                                    budget = max(1, int(self.window_size or 1))
                                    curr = list(window_layers)
                                    prev_only = [x for x in prev if x not in curr]
                                    keep_quota = max(0, budget - len(curr))
                                    keep_tail = (
                                        prev_only[-keep_quota:]
                                        if keep_quota > 0
                                        else []
                                    )
                                    combined = list(keep_tail) + curr
                                    self._recent_windows.append(combined)
                            else:
                                self._recent_windows.append(list(window_layers))
                        else:
                            # Offload / standard eviction
                            self._recent_windows.append(list(window_layers))
                            if int(self._resident_windows) <= 1:
                                old = self._recent_windows.pop(0)
                                try:
                                    self.weight_cache.evict_layers(old)
                                except Exception:
                                    pass
                                try:
                                    self.runtime.model.unload_layers(old)
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
                                            self.weight_cache.evict_layers(old)
                                        except Exception:
                                            pass
                                        try:
                                            if hasattr(
                                                self.runtime.model, "unload_layers"
                                            ):
                                                self.runtime.model.unload_layers(old)
                                                for lid in old:
                                                    self._bound_versions.pop(lid, None)
                                        except Exception:
                                            pass
                    except Exception:
                        pass

                    # continue if next is still local
                    nxt = last_layer + 1
                    if nxt in self.runtime._assigned_set:
                        current_layer = nxt
                        continue

                    # boundary reached
                    x_cast = (
                        x
                        if x.dtype == self.runtime._wire_mx_dtype
                        else x.astype(self.runtime._wire_mx_dtype)
                    )

                    # build output ActivationMessage
                    if nxt >= self.runtime.model_metadata.num_layers:
                        try:
                            with self.runtime._mlx_lock:
                                y = self.runtime.model.normalize(x_cast)
                                y = self.runtime.model.lm_project(y)

                            # Sampling
                            decoding_config = DecodingConfig(
                                temperature=msg.temperature,
                                top_p=msg.top_p,
                                top_k=msg.top_k,
                                repetition_penalty=msg.repetition_penalty,
                                min_p=msg.min_p,
                                min_tokens_to_keep=msg.min_tokens_to_keep,
                            )

                            sampler = Sampler()
                            result = sampler.sample(
                                logits=y,
                                config=decoding_config,
                                req_logprobs=msg.req_logprobs,
                                req_top_logprobs=msg.req_top_logprobs,
                            )

                            token_id = result.token_id
                            token_logprob = result.logprob
                            top_logprobs = result.top_logprobs

                        except Exception as e:
                            logger.error("End-shard sampling failed: %s", e)
                            self.runtime.input_pool.release(msg.pool_id)
                            return

                        output_msg = ActivationMessage(
                            nonce=msg.nonce,
                            layer_id=last_layer,
                            pool_id=-1,
                            shape=cast(tuple[int, ...], x.shape),
                            batch_size=msg.batch_size,
                            timestamp=utc_epoch_now(),
                            node_origin=f"shard_{self.runtime.shard_id}",
                            dtype=str(self.runtime._wire_mx_dtype),
                            callback_url=msg.callback_url,
                            is_final=True,
                            token_id=token_id,
                            logprob=token_logprob,
                            top_logprobs=top_logprobs,
                        )
                    else:
                        output_msg = ActivationMessage(
                            nonce=msg.nonce,
                            layer_id=last_layer,
                            pool_id=-1,
                            shape=cast(tuple[int, ...], x.shape),
                            batch_size=msg.batch_size,
                            timestamp=utc_epoch_now(),
                            node_origin=f"shard_{self.runtime.shard_id}",
                            dtype=str(self.runtime._wire_mx_dtype),
                            callback_url=msg.callback_url,
                            tensor=x_cast,
                            req_logprobs=msg.req_logprobs,
                            req_top_logprobs=msg.req_top_logprobs,
                        )

                    self.runtime.emit_result(output_msg)
                    self.runtime.input_pool.release(msg.pool_id)

                    # Schedule prefetch for next local window if in offload mode
                    if self._mode == "offload":
                        next_window = self._next_local_layers(
                            self.runtime._assigned_sorted, last_layer, self.window_size
                        )
                        loop = self.runtime._loop
                        if loop is None:
                            try:
                                loop = asyncio.get_running_loop()
                            except RuntimeError:
                                logger.error(
                                    "No event loop attached to runtime and none running in thread"
                                )
                                return

                        if next_window is None or len(next_window) == 0:
                            # No next window
                            # prefetch first window for next round for overlap
                            next_window = self.runtime._assigned_sorted[
                                : self.window_size
                            ]
                        fut = loop.run_in_executor(
                            self.runtime.executor,
                            self._prepare_window_blocking,
                            next_window,
                        )
                        self._prepared_by_nonce[msg.nonce] = (next_window, fut)
                    return

        except Exception as e:
            logger.exception("Error in offload policy process: %s", e)
            try:
                if self.runtime.input_pool:
                    self.runtime.input_pool.release(msg.pool_id)
            except Exception:
                pass

    def clear(self):
        for _, fut in self._prepared_by_nonce.values():
            if fut and not fut.done():
                fut.cancel()
        self._prepared_by_nonce.clear()

        try:
            if self.weight_cache:
                self.weight_cache.shutdown()
        except Exception:
            pass

        for layer_id in list(self._bound_versions.keys()):
            try:
                self.weight_cache.evict_layer(layer_id)
            except Exception:
                pass
        try:
            self._bound_versions.clear()
        except Exception:
            self._bound_versions = {}
