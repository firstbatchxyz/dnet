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

        # 1. Local Projections using original weights
        queries = self.base_attn.q_proj(x)
        keys = self.base_attn.k_proj(x)
        values = self.base_attn.v_proj(x)

        # 2. Reshape
        n_heads = self.base_attn.n_heads
        n_kv_heads = self.base_attn.n_kv_heads
        head_dim = self.base_attn.head_dim

        queries = queries.reshape(B, L, n_heads, head_dim)
        keys = keys.reshape(B, L, n_kv_heads, head_dim)
        values = values.reshape(B, L, n_kv_heads, head_dim)

        # 3. RoPE
        # We need to determine the correct offset.
        # If cache is provided, its length implies the offset.
        # But for CP prefill, cache might be None.

        offset = 0
        if cache is not None:
            if isinstance(cache, (list, tuple)):
                # Cache is usually [K, V]
                if cache[0] is not None:
                    offset = cache[0].shape[1]
            elif hasattr(cache, "offset"):
                offset = cache.offset

        if hasattr(self.base_attn, "rope"):
            queries = self.base_attn.rope(queries, offset=offset)
            keys = self.base_attn.rope(keys, offset=offset)

        # 4. Ring Attention via Adapter
        if B != 1:
            logger.warning(f"CP Ring Attention received Batch Size {B} != 1. May fail.")

        q_s = queries.squeeze(0)
        k_s = keys.squeeze(0)
        v_s = values.squeeze(0)

        # Use synchronous wrapper if available, or just call async loop?
        # CPAdapter methods are async. We are in a synchronous MLX forward pass.
        # We need to bridge this.
        # But wait, ShardRuntime.process uses `mx.eval` which blocks?
        # No, `process` is a sync function in `FitInMemoryPolicy`.
        # However, `CPAdapter` uses `asyncio`.

        # CRITICAL ARCHITECTURE ISSUE:
        # `model.forward` is synchronous. `CPAdapter.ring_pass_kv` is async.
        # We cannot await inside `__call__`.
        # We must use `asyncio.run_coroutine_threadsafe` or similar if loop is in another thread?
        # Or `mlx` graph construction is lazy? No, MLX is eager-ish.

        # Solution: The `CPAdapter` must provide a way to execute the ring pass
        # likely by blocking the current thread until the async result is ready.
        # Or `FitInMemoryPolicy` logic needs to be async aware?

        # For v1, let's assume `adapter.ring_pass_kv_attention_sync` handles the bridging.
        # I will implement `ring_pass_kv_attention_sync` in CPAdapter next.

        context_out = self.adapter.ring_pass_kv_attention_sync(q_s, k_s, v_s)

        # 5. Output Projection
        context_out = context_out[None, ...]  # Restore B
        output = self.base_attn.o_proj(context_out.reshape(B, L, -1))

        return output

    def __getattr__(self, name: str):
        if name == "base_attn":
            # Prevent infinite recursion if base_attn is missing
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute 'base_attn'"
            )
        return getattr(self.base_attn, name)
