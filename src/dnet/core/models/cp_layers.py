"""
Context Parallel wrapper layers.
"""

from typing import Optional, Any
import mlx.core as mx
import mlx.nn as nn
from dnet.utils.logger import logger


class CPAttentionWrapper(nn.Module):
    """
    Wraps a standard Attention module to enable Ring Attention.

    Instead of computing local attention, it delegates to the CPAdapter
    to perform distributed Ring Attention (pass-KV or pass-Q).
    """

    def __init__(self, base_attn: nn.Module, adapter: Any):
        super().__init__()
        self.base_attn = base_attn
        self.adapter = adapter

        # Mirror attributes for compatibility
        if hasattr(base_attn, "n_heads"):
            self.n_heads = base_attn.n_heads
        if hasattr(base_attn, "n_kv_heads"):
            self.n_kv_heads = base_attn.n_kv_heads
        if hasattr(base_attn, "head_dim"):
            self.head_dim = base_attn.head_dim
        if hasattr(base_attn, "scale"):
            self.scale = base_attn.scale

        # Debug flag to log weight norms once
        self._weight_logged = False

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """
        Forward pass with Ring Attention injection.
        """
        B, L, D = x.shape

        # Debug: Log input x norm for decode (L==1) at layer 0
        is_decode = L == 1
        if is_decode and hasattr(self.adapter, "current_layer_id"):
            if self.adapter.current_layer_id == 0:
                x_norm = float(mx.sqrt(mx.sum(x**2)))
                x_mean = float(mx.mean(x))
                logger.debug(
                    f"CPAttentionWrapper[L0]: input x_norm={x_norm:.6f}, x_mean={x_mean:.8f}"
                )

                # One-time logging of o_proj weight norm to verify model consistency
                if not self._weight_logged:
                    self._weight_logged = True
                    try:
                        o_proj_w = self.base_attn.o_proj.weight
                        w_norm = float(mx.sqrt(mx.sum(o_proj_w**2)))
                        w_mean = float(mx.mean(o_proj_w))
                        cp_rank = getattr(self.adapter, "rank_id", -1)
                        logger.warning(
                            f"[WEIGHT CHECK] rank={cp_rank} o_proj weight norm={w_norm:.6f}, mean={w_mean:.8f}"
                        )
                    except Exception as e:
                        logger.warning(f"[WEIGHT CHECK] failed: {e}")

        # 1. Local Projections using original weights
        queries = self.base_attn.q_proj(x)
        keys = self.base_attn.k_proj(x)
        values = self.base_attn.v_proj(x)

        # 2. Reshape AND TRANSPOSE to [B, H, L, D] - MUST match mlx-lm order!
        n_heads = self.base_attn.n_heads
        n_kv_heads = self.base_attn.n_kv_heads
        # head_dim may not be directly available on all model architectures (e.g., Qwen3)
        # Fall back to computing from projection output shape
        if hasattr(self.base_attn, "head_dim"):
            head_dim = self.base_attn.head_dim
        else:
            # Compute from q_proj output: queries shape is [B, L, n_heads * head_dim]
            head_dim = queries.shape[-1] // n_heads

        queries = queries.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # 3. RoPE - Applied to [B, H, L, D] format (AFTER transpose!)
        offset = 0
        if cache is not None:
            if hasattr(cache, "offset"):
                offset = cache.offset

        # CP Override: Use global offset from adapter if available
        if hasattr(self.adapter, "current_rope_offset"):
            offset = self.adapter.current_rope_offset

        if hasattr(self.base_attn, "rope"):
            queries = self.base_attn.rope(queries, offset=offset)
            keys = self.base_attn.rope(keys, offset=offset)

        # 4. Ring Attention via Adapter
        if B != 1:
            logger.warning(f"CP Ring Attention received Batch Size {B} != 1. May fail.")

        # Squeeze batch and permute for ring attention: [B, H, L, D] -> [L, H, D]
        # Transpose to [B, L, H, D] then squeeze
        q_s = queries.transpose(0, 2, 1, 3).squeeze(0)  # [L, H, D]
        k_s = keys.transpose(0, 2, 1, 3).squeeze(0)  # [L, H, D]
        v_s = values.transpose(0, 2, 1, 3).squeeze(0)  # [L, H, D]

        # Update Local KV Cache & Retrieve Full Sequence
        k_all = k_s
        v_all = v_s

        if cache is not None:
            # Determine if this is decode (single token) vs prefill (multiple tokens)
            is_decode = L == 1

            # ALL ranks update cache during both prefill and decode.
            # During decode, all ranks store the same decode token to keep caches balanced.
            # The ring_reduce_attention handles deduplication during merge.
            should_update_cache = True

            # 1. Handle MLX Cache Objects (Quantized or Standard)
            if hasattr(cache, "update_and_fetch"):
                if should_update_cache:
                    # MLX cache expects [B, H, L, D] format - keys are already in this format!
                    k_out, v_out = cache.update_and_fetch(keys, values)
                else:
                    # Non-last rank during decode: just fetch without update
                    # For QuantizedKVCache, we need to access the state directly
                    if hasattr(cache, "state") and cache.state is not None:
                        k_out, v_out = cache.state
                    elif hasattr(cache, "keys") and hasattr(cache, "values"):
                        k_out, v_out = cache.keys, cache.values
                    else:
                        # Fallback: use only local K/V
                        k_out, v_out = keys, values

                # Check for quantization (tuple return)
                if isinstance(k_out, tuple):
                    # Dequantize for Ring Attention computation
                    group_size = getattr(cache, "group_size", 64)
                    bits = getattr(cache, "bits", 4)

                    logger.debug(
                        f"CPAttentionWrapper: k_out[0]={k_out[0].shape}, bits={bits}"
                    )

                    k_full = mx.dequantize(
                        k_out[0], k_out[1], k_out[2], group_size, bits
                    )
                    v_full = mx.dequantize(
                        v_out[0], v_out[1], v_out[2], group_size, bits
                    )
                else:
                    # Standard cache (already mx.array)
                    k_full = k_out
                    v_full = v_out

                # Transpose back to [B, L, H, D] and squeeze batch dim for ring attention
                # k_full is [B, H, L, D] -> [B, L, H, D] -> squeeze -> [L, H, D]
                k_all = mx.transpose(k_full, axes=(0, 2, 1, 3)).squeeze(0)
                v_all = mx.transpose(v_full, axes=(0, 2, 1, 3)).squeeze(0)

                # Note: For decode on non-last rank, we do NOT include the new token
                # in k_all/v_all. The new token should only contribute to attention
                # from one shard (last rank) to avoid double-counting during merge.

                logger.debug(f"CPAttentionWrapper: after transpose k_all={k_all.shape}")

            # 2. Handle Simple List Cache (e.g. [K, V])
            elif isinstance(cache, list):
                if cache[0] is not None:
                    if should_update_cache:
                        # keys/values are [B, H, L, D], concatenate on axis=2 (sequence dim)
                        k_c = mx.concatenate([cache[0], keys], axis=2)
                        v_c = mx.concatenate([cache[1], values], axis=2)
                        cache[0] = k_c
                        cache[1] = v_c
                    else:
                        k_c = cache[0]
                        v_c = cache[1]
                    # Transpose to [B, L, H, D] then squeeze
                    k_all = k_c.transpose(0, 2, 1, 3).squeeze(0)
                    v_all = v_c.transpose(0, 2, 1, 3).squeeze(0)
                    # Note: For decode on non-last rank, we do NOT include the new token.

                else:
                    cache[0] = keys
                    cache[1] = values
                    k_all = k_s
                    v_all = v_s

        # Dispatch Logic
        nonce = self.adapter.active_nonce
        layer_id = self.adapter.current_layer_id

        # Use is_decode from earlier (L == 1) - don't redefine it!

        if is_decode:
            # Ring Reduce (Pass-Q/Partial)
            # Efficient for decode where Q is small and KV is distributed
            logger.debug(
                f"CPAttentionWrapper[decode]: q_s={q_s.shape}, k_all={k_all.shape}, v_all={v_all.shape}"
            )
            context_out = self.adapter.ring_reduce_attention_sync(
                q_s,
                k_all,
                v_all,
                rope=self.base_attn.rope,
                nonce=nonce,
                layer_id=layer_id,
            )
        else:
            # Ring Pass-KV
            # Efficient for prefill where KV is sharded and we need All-to-All
            # Note: For prefill, k_all == k_s (chunk)
            context_out = self.adapter.ring_pass_kv_attention_sync(
                q_s,
                k_all,
                v_all,
                rope=self.base_attn.rope,
                nonce=nonce,
                layer_id=layer_id,
            )

        # 5. Output Projection
        context_out = context_out[None, ...]  # Restore B
        output = self.base_attn.o_proj(context_out.reshape(B, L, -1))

        # Debug: Log final attention output for decode at layer 0
        if is_decode and hasattr(self.adapter, "current_layer_id"):
            if self.adapter.current_layer_id == 0:
                out_norm = float(mx.sqrt(mx.sum(output**2)))
                out_mean = float(mx.mean(output))
                logger.debug(
                    f"CPAttentionWrapper[L0]: OUTPUT norm={out_norm:.6f}, mean={out_mean:.8f}"
                )

        return output

    @property
    def q_proj(self):
        return self.base_attn.q_proj

    @property
    def k_proj(self):
        return self.base_attn.k_proj

    @property
    def v_proj(self):
        return self.base_attn.v_proj

    @property
    def o_proj(self):
        return self.base_attn.o_proj

    @property
    def rope(self):
        return getattr(self.base_attn, "rope", None)
