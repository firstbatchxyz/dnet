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
        return partials[0].output

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
    Merge two partial attention outputs using online softmax algorithm.

    This is the core operation for ring reduction - allows progressive
    merging without All2All.

    Args:
        a: First partial output
        b: Second partial output

    Returns:
        Merged partial output (can be merged again with more partials)
    """
    # Find new max for numerical stability
    m_new = mx.maximum(a.max_score, b.max_score)

    # Compute scaling factors
    # exp(m_old - m_new) to rescale old values
    scale_a = mx.exp(a.max_score - m_new)
    scale_b = mx.exp(b.max_score - m_new)

    # Rescale log-sum-exp values
    l_a_scaled = scale_a * a.log_sum_exp
    l_b_scaled = scale_b * b.log_sum_exp
    l_new = l_a_scaled + l_b_scaled

    # Avoid division by zero
    l_new_safe = mx.where(l_new == 0, mx.ones_like(l_new), l_new)

    # Merge outputs with proper weighting
    # Need to expand dims for broadcasting with output tensor
    # output shape: [..., heads, dim], scales shape: [..., heads]
    scale_a_expanded = mx.expand_dims(scale_a, axis=-1)
    scale_b_expanded = mx.expand_dims(scale_b, axis=-1)
    l_a_expanded = mx.expand_dims(l_a_scaled, axis=-1)
    l_b_expanded = mx.expand_dims(l_b_scaled, axis=-1)
    l_new_expanded = mx.expand_dims(l_new_safe, axis=-1)

    output_new = (
        scale_a_expanded * l_a_expanded * a.output
        + scale_b_expanded * l_b_expanded * b.output
    ) / l_new_expanded

    return PartialAttentionOutput(
        output=output_new,
        max_score=m_new,
        log_sum_exp=l_new,
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

    return PartialAttentionOutput(
        output=output,
        max_score=max_score,
        log_sum_exp=sum_exp,
    )
