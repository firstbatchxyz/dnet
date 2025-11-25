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


@register_policy("fit")
class FitInMemoryPolicy(ComputePolicy):
    """Everything fits - no offloading needed"""

    def configure_policy_for_model(self, req: ShardLoadModelRequest) -> None:
        self._mode = "fit"
        local_count = max(1, len(self.runtime.assigned_layers))
        requested_w = max(1, int(req.window_size))
        self.window_size = min(local_count, requested_w)
        self.weight_cache = WeightCache(
            self.runtime.assigned_layers,
            self.runtime.model_metadata,
            window_size=self.window_size,
            prefetch_threads=self.runtime.prefetch_threads,
            resident_windows=self._resident_windows,
            use_mxload_fastpath=self.runtime.compute_config.mxload_fastpath,
            prefetch_mode=self.runtime.compute_config.prefetch_mode,
        )

    def process(self, msg: ActivationMessage) -> None:
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

                # 1) per-nonce KV
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
                    # build contiguous window inside our shard
                    window_layers: list[int] = []
                    for i in range(self.window_size):
                        layer = current_layer + i
                        if layer not in self.runtime._assigned_set:
                            break
                        window_layers.append(layer)

                    to_bind = self._bind_layer_weights(window_layers, msg)
                    if to_bind is None:
                        return
                    if to_bind:
                        self.runtime._compute_busy.set()
                        with self.runtime._mlx_lock:
                            self.runtime.model.load_weights(
                                list(to_bind.items()), strict=False
                            )

                    # compute window
                    try:
                        self.runtime._compute_busy.set()
                    except Exception:
                        pass
                    for lyr in window_layers:
                        with self.runtime._mlx_lock:
                            x = self.runtime.model.apply_single_layer(lyr, x, cache=kv)
                            try:
                                if str(x.dtype) != str(self.runtime._wire_mx_dtype):
                                    x = x.astype(self.runtime._wire_mx_dtype)
                            except Exception:
                                pass

                    last_layer = window_layers[-1]
                    try:
                        mx.eval(x)
                    except Exception:
                        pass

                    for lid in window_layers:
                        self.weight_cache.decrease_reference(lid)

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
                        # end-shard sampling
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
                    return

        except Exception as e:
            logger.exception("Error in fit policy process: %s", e)
            try:
                if self.runtime.input_pool:
                    self.runtime.input_pool.release(msg.pool_id)
            except Exception:
                pass

    def clear(self):
        try:
            if self.weight_cache:
                self.weight_cache.cancel_all_prefetch()
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
