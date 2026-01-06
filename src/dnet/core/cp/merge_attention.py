"""Merge attention operator for context parallelism.

When computing blockwise attention across distributed KV caches, each device
produces partial outputs with local softmax statistics. These must be merged
correctly using numerically stable rescaling.

Math:
    For blocks with outputs O_i, max scores m_i, and log-sum-exp l_i:
    m_global = max(m_1, m_2, ..., m_N)
    l_global = sum(exp(m_i - m_global) * l_i)
    O_merged = sum(exp(m_i - m_global) * l_i * O_i) / l_global
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class PartialAttentionOutput:
    """Partial attention output with merge statistics.

    Attributes:
        output: Attention output [batch, seq, heads, dim] or [seq, heads, dim]
        max_score: Per-position max attention score [batch, seq, heads] or [seq, heads]
        log_sum_exp: Per-position log-sum-exp of attention weights (same shape as max_score)
    """

    output: mx.array
    max_score: mx.array
    log_sum_exp: mx.array


def merge_partial_attention(
    partials: list[PartialAttentionOutput],
) -> mx.array:
    """
    Merge multiple partial attention outputs with numerically stable rescaling.

    This implements the online softmax merge algorithm from Flash Attention,
    extended for distributed computation.

    Args:
        partials: List of partial outputs from different KV blocks/ranks

    Returns:
        Merged attention output tensor
    """
    if not partials:
        raise ValueError("Cannot merge empty list of partials")

    if len(partials) == 1:
        # Single partial: still need to normalize since output is unnormalized
        sum_exp_expanded = mx.expand_dims(partials[0].log_sum_exp, axis=-1)
        return partials[0].output / sum_exp_expanded

    # Start with first partial as running state
    running = partials[0]

    for partial in partials[1:]:
        running = merge_two_partials(running, partial)

    return running.output


def merge_two_partials(
    a: PartialAttentionOutput,
    b: PartialAttentionOutput,
) -> PartialAttentionOutput:
    """
    Merge two partial attention outputs using numerically stable sigmoid-based algorithm.

    This implements the merge formula from ring-flash-attention which uses sigmoid
    and logsigmoid to keep values bounded and prevent numerical explosion:
        out = out - sigmoid(block_lse - lse) * (out - block_out)
        lse = lse - logsigmoid(lse - block_lse)

    Reference: https://github.com/zhuzilin/ring-flash-attention/pull/34

    Args:
        a: First partial output (running state)
        b: Second partial output (new block to merge)

    Returns:
        Merged partial output
    """
    # Convert to float32 for numerical precision (matching reference)
    out_a = a.output.astype(mx.float32)
    out_b = b.output.astype(mx.float32)
    lse_a = a.log_sum_exp.astype(mx.float32)
    lse_b = b.log_sum_exp.astype(mx.float32)
    # Sigmoid-based merge (bounded, numerically stable)
    # sigmoid(x) = 1 / (1 + exp(-x))
    # out = out_a - sigmoid(lse_b - lse_a) * (out_a - out_b)

    # Expand lse for broadcasting with output [S_q, H, D]
    lse_a_exp = mx.expand_dims(lse_a, axis=-1)
    lse_b_exp = mx.expand_dims(lse_b, axis=-1)

    # sigmoid(lse_b - lse_a) - bounded between 0 and 1
    sig = mx.sigmoid(lse_b_exp - lse_a_exp)

    # Merge outputs: out = out_a - sig * (out_a - out_b) = out_a * (1 - sig) + out_b * sig
    output_new = out_a - sig * (out_a - out_b)

    # Update LSE using logsigmoid
    # lse = lse_a - logsigmoid(lse_a - lse_b)
    # logsigmoid(x) = -log(1 + exp(-x)) = x - log(1 + exp(x)) for numerical stability
    # lse_new = lse_a - logsigmoid(lse_a - lse_b)
    #         = lse_a + log(1 + exp(lse_b - lse_a))  [using -logsigmoid(x) = log(1 + exp(-x))]
    #         = lse_a + softplus(lse_b - lse_a)
    # Or equivalently: max(lse_a, lse_b) + log(1 + exp(-|lse_a - lse_b|))
    # Which is the stable log-sum-exp of two values
    lse_max = mx.maximum(lse_a, lse_b)
    lse_new = lse_max + mx.log(
        mx.exp(lse_a - lse_max) + mx.exp(lse_b - lse_max) + 1e-10
    )

    return PartialAttentionOutput(
        output=output_new,
        max_score=lse_max,  # Keep for compatibility
        log_sum_exp=lse_new,
    )


def compute_partial_attention_stats(
    attention_weights: mx.array,
    values: mx.array,
) -> PartialAttentionOutput:
    """
    Compute attention output with statistics needed for merging.

    This should be called after computing raw attention scores but before
    the final softmax normalization.

    Args:
        attention_weights: Raw attention scores [batch, heads, seq_q, seq_kv]
        values: Value tensor [batch, seq_kv, heads, dim]

    Returns:
        PartialAttentionOutput with output and merge statistics
    """
    # Get max for numerical stability
    max_score = mx.max(attention_weights, axis=-1)  # [batch, heads, seq_q]

    # Compute softmax with numerical stability
    shifted = attention_weights - mx.expand_dims(max_score, axis=-1)
    exp_weights = mx.exp(shifted)
    sum_exp = mx.sum(exp_weights, axis=-1)  # [batch, heads, seq_q]

    # Normalize
    normalized = exp_weights / mx.expand_dims(sum_exp, axis=-1)

    # Compute attention output
    # normalized: [batch, heads, seq_q, seq_kv]
    # values transposed: [batch, heads, seq_kv, dim]
    values_transposed = mx.transpose(values, (0, 2, 1, 3))
    output = mx.matmul(normalized, values_transposed)  # [batch, heads, seq_q, dim]

    # Transpose output back to [batch, seq_q, heads, dim]
    output = mx.transpose(output, (0, 2, 1, 3))

    # Transpose stats to match output: [batch, seq_q, heads]
    max_score = mx.transpose(max_score, (0, 2, 1))
    sum_exp = mx.transpose(sum_exp, (0, 2, 1))

    # Compute proper log-sum-exp: LSE = max + log(sum_exp)
    lse = max_score + mx.log(sum_exp + 1e-10)

    return PartialAttentionOutput(
        output=output,
        max_score=max_score,
        log_sum_exp=lse,
    )
