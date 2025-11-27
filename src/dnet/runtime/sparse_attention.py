
import time
import math
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
#from ..util import logger

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3 import ModelArgs
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.base import scaled_dot_product_attention 

logger = logging.getLogger(__name__)

BLOCK_SIZE = 128 
#REDUCTION_KERNEL_SIZE = 2

# NOTE Look into AdaFlash for adaptive fine-grained 

# NOTE: Rely on mx.matmul for dense tile sgemm for now, until we have metal kernels 
# Compute sparsity of packed matrices per block
def sparse_dot_product_attention_blocked(
  Q: mx.array,     # (B, Hq,  Lq, D)
  K_sel: mx.array, # (B, Hkv, Lk_sel_max, D) (packed)
  V_sel: mx.array, # (B, Hkv, Lk_sel_max, D) (packed)
  scale: float,
  mask: Optional[mx.array] = None,  # (B, Hq, Lq, Lk_sel_max)
  padding_mask: Optional[mx.array] = None,  # (B, Hq, 1, 1)
  selected_counts: Optional[mx.array] = None,
  group_size: int = 64,
  bits: int = 8
) -> mx.array:

  K = K_sel
  V = V_sel
  B, n_q_heads, L, D = Q.shape
  n_kv_heads = K.shape[-3]
  n_repeats = n_q_heads // n_kv_heads

  #print(Q.shape, K.shape, V.shape)
  Q *= scale

  if n_repeats > 1:
    Q = mx.reshape(Q, (B, n_kv_heads, n_repeats, L, D)) 
  else:
    Q = mx.expand_dims(K, axis=-3)
  K = mx.expand_dims(K, axis=-3)
  V = mx.expand_dims(V, axis=-3)

  scores = mx.matmul(Q, K.transpose(0, 1, 2, 4, 3))   
  #print(scores)
  if mask is not None:
    """ # Use for running non-sparse attention
    if padding_mask is not None:
      padding_mask = padding_mask.reshape(1, -1)
      padding_mask = mx.repeat(padding_mask, L, axis=0)
      scores = mx.where(padding_mask, scores, mx.finfo(scores.dtype).min)
    """

    Lq, Lkv = scores.shape[-2:]
    if Lq > 1:
      q_idx = mx.arange(Lq)
      k_idx = mx.arange(Lkv)
      mask = q_idx.reshape(-1, 1) >= k_idx.reshape(1, -1)
      #if mask.ndim > 2:
      #  mask = mask.reshape(B, n_kv_heads, 1, mask.shape[2], mask.shape[3])
      scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        

    #print(mask)
  scores = mx.softmax(scores, axis=-1, precise=True)
  out = mx.matmul(scores, V)   

  if n_repeats > 1:
      out = mx.reshape(out, (B, n_q_heads, L, D)) 

  return out.transpose(0, 2, 1, 3).reshape(B, L, -1) 


@dataclass
class StrategyInput:
  Q: mx.array 
  K: mx.array
  mask: Optional[mx.array]
  block_size: int                   # pooling block size
  gqa_interleave: bool = False      # mapping style
  last_q: Optional[mx.array] = None # (B, block_size, Hq, D)
  gamma: float = 0.9                # top-p value [0, 1]
  min_budget: int = 1               # min blocks nr
  max_budget: int = 2147483647      # max blocks nr 
  tau: float = 0.0                  # JSD threashold
  update_frequency: int = 10        # # cycles before we refresh JSD metrics
  is_token_level: bool = False      # Token-level sparsity indices


@dataclass
class Strategy:
  blocks: mx.array 
  tokens: mx.array
  block_size: int
  selected_counts: Optional[mx.array]
  min_budget: int
  max_budget: int
  update_frequency: int = 10
  # TODO: Debug strategy metadata
  # Potentially create different budgets?


class AbstractSparseStrategy:
  """ Strategies decide the block indices of attention we compute """

  def __init__(self, strc: Dict[str, Any] = None) -> None:
    pass

  def reset(self) -> None:
    pass

  def __call__(self, input: StrategyInput) -> Strategy: 
    pass

  def update(self, input: StrategyInput, prev_blocks: mx.array, prev_Lkv: int) -> Strategy:
    """ Update the indices every decode loop for new tokens """ 
    pass


# https://arxiv.org/abs/2502.20766
class FlexPrefillSparseAttention(AbstractSparseStrategy):
  def __init__(self, strc: Dict[str, Any] = None):
    super().__init__()
    self.blocks = None 
    self.score_cache = None # for incremental update
    self.pattern_type = None # Query-Aware (0) or Vertical-Slash (1)

  def reset(self):
    self.blocks = None
    self.score_cache = None
    self.pattern_type = None

  def kullback_leibler(self, d0, d1):
    return mx.sum(d0 * mx.log2(d0 / d1))

  def jensen_shannon_metric(self, d0, d1):
    median = 0.5*(d0 + d1)
    div = 0.5*self.kullback_leibler(d0, median) + 0.5*self.kullback_leibler(d1, median)
    return div**0.5

  def top_p(self, x, p:float):
    cum_probs = mx.cumsum(x, axis=1)
    top = mx.where(cum_probs > 1-p, x, mx.zeros_like(x))
    return top

  # Q:   (B, Lq,  Hkv, D)
  # K/V: (B, Lkv, Hkv, D)
  # Computes a low-resolution map of sparse indices depending on the input query
  def query_aware_search(self, Q, K, gamma, head_dim, min_budget, max_budget, prev_scores=None, prev_Lkv=None):
    assert min_budget >= 1 and max_budget >= 1, "budgets must be at least 1"
    assert min_budget <= max_budget, "min_budget must be smaller then max_budget"
    pool = nn.AvgPool1d(kernel_size=BLOCK_SIZE, stride=BLOCK_SIZE)

    D = head_dim
    B, Lq, Hq = Q.shape[:3]
    Lk = K.shape[1]
    Hkv = K.shape[2]
    n_repeats = Hq // Hkv
    assert Hq % Hkv == 0, "query heads must be a multiple of KV heads for GQA"
    num_blocks = math.ceil(Lk / BLOCK_SIZE)

    # Group Q for GQA
    if Lq < BLOCK_SIZE:
      padding = BLOCK_SIZE - Lq 
      Q = mx.pad(Q, [(0,0),(0,padding), (0,0), (0,0)], constant_values=0)
    Q_gqa = Q[:, -BLOCK_SIZE:, :, :]
    Q_gqa = Q_gqa.reshape(B, BLOCK_SIZE, n_repeats, Hkv, head_dim)
    Q_gqa = mx.mean(Q_gqa, axis=2) # (B, BLOCK_SIZE, Hkv, head_dim)

    # TODO: Handle the step % repeat_count condition for recomputation too
    if prev_scores is not None and prev_Lkv is not None: # Only compute new blocks 
      new_blocks = num_blocks - math.ceil(prev_Lkv / BLOCK_SIZE) 
      if new_blocks > 0:
        K_new = K[:, prev_Lkv:, :]
        L_new = Lk - prev_Lkv 
        assert L_new > 0
        K_hat_new = pool(K_new.reshape(B*Hkv, L_new, D)).reshape(B, Hkv, -1, D)
        A_hat_new = nn.softmax(mx.matmul(Q_gqa.reshape(B*Hkv, Lq, D), K_hat_new.transpose(0, 1, 3, 2)) / mx.sqrt(D), axis=-1)
        A_hat_new = (A_hat_new / mx.sum(A_hat_new, axis=-1, keepdims=True)).reshape(B, Hkv, -1) #(B, Hkv, new_blocks)
        A_hat = mx.concatenate([prev_scores, A_hat_new], axis=-1)
        assert A_hat.shape[-1] == num_blocks, "Invalid number of blocks selected in A_hat."

      else: # No new blocks
        A_hat = prev_scores 

    else: # Full compute
      Q_hat = pool(Q_gqa.reshape(B*Hkv, BLOCK_SIZE, D)).reshape(B, Hkv, -1, D)
      K_hat = pool(K.reshape(B*Hkv, Lk, D)).reshape(B, Hkv, -1, D)
      A_hat = nn.softmax(mx.matmul(Q_hat, K_hat.transpose(0, 1, 3, 2)) / mx.sqrt(D), axis=-1) 
      A_hat = A_hat / mx.sum(A_hat, axis=-1, keepdims=True)
      A_hat = mx.mean(A_hat, axis=-2)
      assert A_hat.shape[-1] == num_blocks, "Invalid number of blocks selected in A_hat."

    # Pluck selected values 
    # Use argmax to get the index of the first True value, where the threshold was met
    # then handle edge cases where the threshold is never met by selecting all blocks
    self.score_cache = A_hat
    I_a = mx.argsort(A_hat, axis=-1)[..., ::-1] # (B, Hkv, num_blocks)
    active_blocks = mx.take_along_axis(A_hat, I_a, axis=-1)
    csum = mx.cumsum(active_blocks, axis=-1) # (B, Hkv, num_blocks)

    target = gamma * mx.sum(A_hat, axis=-1, keepdims=True) if 0 <= gamma <= 1.0 else gamma
    selected_counts = mx.argmax(csum >= target, axis=-1) + 1  # (B, Hkv), reduce over num_blocks
    full_sum = mx.sum(A_hat, axis=-1) # (B, Hkv)
    selected_counts = mx.where(full_sum < target.squeeze(-1), num_blocks, selected_counts)
    #TODO: FIX target degrading due to softmax

    # Apply min/max budget per head
    selected_counts = mx.maximum(mx.minimum(selected_counts, max_budget), min_budget)
    selected_counts = mx.minimum(selected_counts, num_blocks)

    # Pad and format to top selected_counts indices per head (sorted)
    #max_sel = max_budget
    max_sel = int(selected_counts.max()) 
    blocks = mx.full([B, Hkv, max_sel], -1, dtype=mx.int32)
    for b in range(B):
      for h in range(Hkv):
        count = int(selected_counts[b, h])
        selected_idxs = I_a[b, h, :count]
        selected_idxs = mx.sort(selected_idxs)
        actual_count = min(selected_idxs.shape[0], max_sel)
        blocks[b, h, :actual_count] = selected_idxs[:actual_count]

    return blocks, selected_counts
    

  # VS Index search
  # Token-level sparsity
  def vertical_slash_search(self, Q, K, gamma, D, min_budget, max_budget, prev_scores=None, prev_Lkv=None):
    assert min_budget >= 1 and max_budget >= 1, "budgets must be at least 1"
    assert min_budget <= max_budget, "min_budget must be smaller then max_budget"
    B, Lq, Hq = Q.shape[:3]
    Hkv = K.shape[2]
    n_repeats = Hq // Hkv
    Lkv = K.shape[1]

    # Group Q for GQA
    Q_gqa = Q[:, -BLOCK_SIZE:, :, :]
    Q_gqa = Q.reshape(B, Lq, n_repeats, Hkv, D)
    Q_gqa = mx.mean(Q_gqa, axis=2)  # (B, Lq, Hkv, D)
    #print(Q_gqa.shape)


    # Compute a_v vertical sum and a_s diagonal sum
    # NOTE: Maintain batch and head independence for each sum
    # NOTE: Normalization happens after the raw values are partially or fully computed
    if prev_scores is not None and prev_Lkv is not None: # Only compute the update to sequence length
      new_cols = Lkv - prev_Lkv 
      if new_cols > 0:
        prev_a_v, prev_a_s = prev_scores
        new_K = K[:, prev_Lkv:, :]
        # TODO: Broadcasting on A_hat_new is most likely broken. Check full compute.
        A_hat_new = nn.softmax(mx.matmul(Q_gqa, new_K.transpose(0, 1, 3, 2)) / mx.sqrt(D), axis=-1)
        new_a_v = mx.sum(A_hat_new, axis=1) 
        new_a_s = mx.zeros_like(prev_a_s) + prev_a_s

        offset_start = -(Lq-1) 
        num_diags = Lq + Lkv -1
        assert new_a_s.shape[-1] == Lq + prev_Lkv - 1
        if num_diags > new_a_s.sape[-1]:
          new_a_s = mx.pad(new_a_s, [(0,0), (0,0), (0,num_diags - new_a_s.shape[-1])], constant_values=0)

        for b in range(B):
          for h in range(Hkv):
            for off in range(prev_Lkv, Lkv):
              if off >= prev_Lkv - Lq + 1:
                diag_val = mx.trace(A_hat_new[b, :, h, :], offset=off-prev_Lkv)
                new_a_s[b, h, off - offset_start] = diag_val

        a_v = mx.concatenate([prev_a_v, new_a_v], axis=-1)
        a_s = mx.concatenate([prev_a_s, new_a_s], axis=-1)
      else:
        a_v, a_s = prev_scores

    else: # Full compute
      # Collapse BxHkv because MLX broadcasting sucks 
      Q_flat = Q_gqa.reshape(B*Hkv, Lq, D)
      K_flat = K.reshape(B*Hkv, Lkv, D)
      A_hat = nn.softmax(mx.matmul(Q_flat, K_flat.transpose(0, 2, 1)) / mx.sqrt(D), axis=-1)
      A_hat = A_hat.reshape(B, Hkv, Lq, Lkv).transpose(0, 2, 1, 3)

      a_v = mx.sum(A_hat, axis=1) 
      a_s = mx.zeros([B, Hkv, Lq + Lkv -1]) # Diagonal
      offset_start = -(A_hat.shape[1]-1)
      for b in range(B):
        for h in range(Hkv):
          for off in range(offset_start, A_hat.shape[-1]):
            diag_val = mx.trace(A_hat[b, :, h, :], offset=off)
            a_s[b, h, off - offset_start] = diag_val

      # Store cache before normalization 
      self.score_cache = (a_v, a_s)

      # Normalize
      total = mx.sum(a_v, axis=-1, keepdims=True) # Already normalized over 1(Lq)
      a_v = a_v / mx.maximum(total, 1e-6) # Avoid div by 0
      a_s = a_s / mx.maximum(mx.sum(a_s, axis=-1, keepdims=True), 1e-6)

      # Per head selection
      I_v = mx.argsort(a_v, axis=-1)[::-1]
      I_s = mx.argsort(a_s, axis=-1)[::-1]
      #print(I_v, I_s)
      plucked_v = mx.cumsum(mx.take_along_axis(a_v, I_v, axis=-1), axis=-1)
      plucked_s = mx.cumsum(mx.take_along_axis(a_s, I_s, axis=-1), axis=-1)

      target_v = gamma * mx.sum(a_v, axis=-1, keepdims=True) if 0 <= gamma <= 1.0 else gamma
      target_s = gamma * mx.sum(a_s, axis=-1, keepdims=True) if 0 <= gamma <= 1.0 else gamma
      K_v = mx.argmax(plucked_v >= target_v, axis=-1) + 1
      K_s = mx.argmax(plucked_s >= target_s, axis=-1) + 1
      full_sum_v = mx.sum(a_v, axis=-1) # (B, Hkv)
      full_sum_s = mx.sum(a_s, axis=-1) # (B, Hkv)
      K_v = mx.where(full_sum_v < target_v.squeeze(-1), a_v.shape[-1], K_v)
      K_s = mx.where(full_sum_s < target_s.squeeze(-1), a_s.shape[-1], K_s)

      b# Concat, sort, unique per head
      max_sel = min( int((K_v + K_s).max()), max_budget) # Worst case number
      blocks = mx.full([B, Hkv, max_budget], -1, dtype=mx.int32)
      for b in range(B):
        for h in range(Hkv):
          sel_v = I_v[b, h, :int(K_v[b, h].item())]
          sel_s = I_s[b, h, :int(K_s[b, h].item())]
          selected = mx.sort(mx.concatenate([sel_v, sel_s]))

          # WARNING: VERY SLOW
          # No unique op in mlx, because we are merging both sel_v and sel_s we need it
          # TODO: When MLX adds support for boolean indices change this.
          if selected.size > 0:
            diffs = mx.concatenate([mx.array([True]), selected[1:] != selected[:-1]]) # Filter duplicates
            unique_idxs = [0]
            for i in range(1, selected.size):
              if selected[i] != selected[unique_idxs[-1]]:
                unique_idxs.append(i)
            selected = selected[mx.array(unique_idxs, dtype=mx.int32)]
          num_selected = max(min(selected.size, max_budget), min_budget)
          num_selected = min(num_selected, selected.size)
          blocks[b, h, :num_selected] = selected[:num_selected]

      return blocks


  # 0 - query specific, 1 - vertical slash
  # NOTE: Causal mask is not applied when comparing distributions 
  def sparse_pattern_search(self, Q, K, tau, BLOCK_SIZE):
    assert K.shape[1] % BLOCK_SIZE == 0
    pool = nn.AvgPool1d(kernel_size=BLOCK_SIZE, stride=BLOCK_SIZE)
    B, Lq, Hq, D = Q.shape
    Lkv = K.shape[1]
    Hkv = K.shape[2]
    n_repeats = Hq // Hkv # GQA repeats

    # Pad repersentative subset
    if Lq < BLOCK_SIZE:
      padding = BLOCK_SIZE - Lq
      rep = Q[:, -Lq:, :, :]
      rep = mx.pad(rep, [(0,0), (0, padding), (0,0), (0,0)])
      Lq = BLOCK_SIZE 
    else:
      rep = Q[:, -BLOCK_SIZE:, :, :]

    if n_repeats > 1: # mean over n_repeats (rep size becomes Hkv)
      rep = rep.reshape(B, BLOCK_SIZE, n_repeats, Hkv, D)
      rep_group = mx.mean(rep, axis=2).transpose(0, 2, 1, 3) # (B, Hkv, BLOCK_SIZE, D)
    else:
      rep_group = rep

    Q_est = pool(rep_group.reshape(B*Hkv, BLOCK_SIZE, D)).reshape(B, Hkv, -1, D)
    K_est = pool(K.reshape(B*Hkv, K.shape[1], D)).reshape(B, Hkv, -1, D)
    a_est = nn.softmax(mx.matmul(Q_est, K_est.transpose(0, 1, 3, 2)) / mx.sqrt(D), axis=-1) 

    if n_repeats > 1: # (B, Hkv, n_repeat, BLOCK_SIZE, D) @ (B, Hkv, D, Lq)
      K = K.reshape(B, Lkv, Hkv, 1, D)
      a_true = mx.matmul(rep.transpose(0, 3, 2, 1, 4), K.transpose(0, 2, 3, 4, 1)) / mx.sqrt(D)
      a_true = mx.mean(a_true, axis=2)
      a_true = nn.softmax(a_true, axis=-1)
      a_true = pool(a_true.reshape(B*Hkv, BLOCK_SIZE, -1)).reshape(B, Hkv, Lkv, -1 ) 
    else:
      a_true = nn.softmax(mx.matmul(rep.transpose(0, 2, 1, 3), K.transpose(0, 2, 3, 1)) / mx.sqrt(D), axis=-1)
      a_true = pool(a_true.reshape(B*Hkv, BLOCK_SIZE, -1)).reshape(B, Hkv, -1, ) 

    djs = mx.zeros([B, Hkv])
    for b in range(B):
      for h in range(Hkv):
        djs_bh = self.jensen_shannon_metric(a_est[b, h], a_true[b, h])
        djs[b, h] = mx.mean(djs_bh) # If multi-dim
    return mx.where(djs < tau, 1, 0) # 1 for vertical-slash, 0 for query-aware

  # Analyze heads and create sparse strategy
  def __call__(self, i: StrategyInput) -> Strategy:
    Q, K, gamma, tau, block_size = i.Q, i.K, i.gamma, i.tau, i.block_size 
    min_budget, max_budget = i.min_budget, i.max_budget
    B, Lq, Hq, D = Q.shape
    _, Lkv, Hkv, _ = K.shape
    self.pattern_type = self.sparse_pattern_search(Q, K, tau, block_size)
    B, Hkv = self.pattern_type.shape

    """
    for b in range(B):
      for h in range(Hkv):
        #if self.pattern_type[b, h] == 0:
        if True: # Hardcode query_aware for now
          sel, selected_counts = self.query_aware_search(Q, K, gamma, D, i.min_budget, i.max_budget)
          blocks[b, h, :sel.shape[-1]] = sel[b, h]
        else:
          sel = self.vertical_shash_search(Q, K, gamma, D, i.min_budget, i.max_budget)
          blocks[b, h, :sel.shape[0]] = sel
    """

    selected_counts = mx.zeros([B, Hkv])
    #pttn = self.pattern_type[b][h].item()
    pttn = 0
    if pttn == 0:
      sel, selected_counts = self.query_aware_search(Q, K, gamma, D, min_budget, max_budget)
    else:
      sel = self.vertical_shash_search(Q, K, gamma, head_dim, min_budget, max_budget)

    max_sel = int(mx.max(selected_counts).item())
    blocks = mx.full([B, Hkv, max_sel], -1, dtype=mx.int32)
    for b in range(B):
      for h in range(Hkv):
        count = int(selected_counts[b][h].item())
        blocks[b, h, :count] = sel[b][h][:count]

    return Strategy(blocks=blocks, 
      tokens=None, 
      block_size=BLOCK_SIZE, 
      selected_counts=selected_counts,
      max_budget=i.max_budget,
      min_budget=i.min_budget)


  def update(self, i: StrategyInput, prev_S: Strategy, prev_Lkv:int, cycles) -> Strategy:
    Q, K, gamma, tau, block_size = i.Q, i.K, i.gamma, i.tau, i.block_size 
    min_budget, max_budget = prev_S.min_budget, prev_S.max_budget
    B, Lq, Hkv, D = Q.shape
    Lk = K.shape[1]

    if cycles % prev_S.update_frequency != 0:
      return prev_S

    # Recompute patterns only on new blocks (maybe add a set repeat pattern)
    new_blocks = math.ceil(Lk / block_size) - math.ceil(prev_Lkv / block_size)
    if new_blocks > 0 or self.pattern_type is None:
      self.pattern_type = self.sparse_pattern_search(Q, K, tau, block_size)

    # Update per head (recompute all for now)
    selected_counts = mx.zeros([B, Hkv])
    #pttn = self.pattern_type[b][h].item()
    pttn = 0
    if pttn == 0:
      sel, selected_counts = self.query_aware_search(Q, K, gamma, D, min_budget, max_budget)
    else:
      sel = self.vertical_shash_search(Q, K, gamma, head_dim, min_budget, max_budget)

    max_sel = int(mx.max(selected_counts).item())
    blocks = mx.full([B, Hkv, max_sel], -1, dtype=mx.int32)
    for b in range(B):
      for h in range(Hkv):
        count = int(selected_counts[b][h].item())
        blocks[b, h, :count] = sel[b][h][:count]

    return Strategy(
      blocks=blocks, 
      tokens=None, 
      block_size=BLOCK_SIZE, 
      selected_counts=selected_counts,
      max_budget=max_budget,
      min_budget=min_budget)

class SparseAttention(nn.Module):
  def __init__(self, args: ModelArgs, sparse_strategy: AbstractSparseStrategy, prefill=False):
    super().__init__()

    h = args.hidden_size
    self.n_heads = args.num_attention_heads
    self.n_kv_heads = args.num_key_value_heads
    self.head_dim = args.head_dim
    self.scale = args.head_dim**-0.5
    self.S = None # Keep a cached S and only update K/V L_kv dimension on decode 
    self.compute_strat = sparse_strategy
    self.kv_cache = None
    self.sparse_kv_cache = None
    self.cycles = 0

    self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(self.n_heads*self.head_dim, h, bias=False)

    self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
    self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
    self.rope = initialize_rope(
      self.head_dim,
      base=args.rope_theta,
      traditional=False,
      scaling_config=None,
      max_position_embeddings=40960,
    )

  # Returns a flat, sorted list of absolute token indices for the blocks. 
  # Expects KV cache in (B, Hkv, Nblk, blk, D) format. 
  def blocks_to_positions(self, block_ids, blk, Lk):
    pos = []
    for id in block_ids:
      start = id*blk
      end = min(start + blk, Lk)
      if start < end:
        pos.extend(range(start, end))
    return pos

  # K_bh: (Nblk, blk, D)
  # Return: (Lk_sel, D)
  def gather_blocks(self,K_bh, blocks, blk):
    out = []
    for id in blocks:
      slab = K_bh[id]
      out.append(slab)
    return mx.concatenate(out, axis=0)

  # K, V: (B, Hkv, Nblk, blk, D)
  # S: (B, Hkv) (sorted) 
  # NOTE: Also computes the causal mask
  def pack_kv_blocks(self, K, V, S, Lkv, Lq, uniform=True, padding_mask=None):
    _, _, sel_max = S.blocks.shape
    B, Hkv, Lkv_total, D = K.shape
    #print(Lkv_total, Lkv)
    assert Lkv_total == Lkv, "Cache length mismatch"
    num_blocks = math.ceil(Lkv / S.block_size)
    K_sel = mx.full([B, Hkv, sel_max, S.block_size, D], 0.0) 
    V_sel = mx.full([B, Hkv, sel_max, S.block_size, D], 0.0) 

    if padding_mask is None:
      padding_mask = mx.full([Lkv], 0.0, dtype=K.dtype)

    K = K.reshape(B, Hkv, num_blocks, S.block_size, D)
    V = V.reshape(B, Hkv, num_blocks, S.block_size, D)
    padding_mask = padding_mask.reshape(num_blocks, S.block_size, 1)
    mask = mx.zeros([B, Hkv, Lq, sel_max*S.block_size], dtype=mx.bool_) 

    for b in range(B):
      for h in range(Hkv):
        K_sel_b = mx.full([sel_max, S.block_size, D], 0.0) 
        V_sel_b = mx.full([sel_max, S.block_size, D], 0.0) 
        mask_b = mx.zeros([Lq, sel_max * S.block_size], dtype=mx.bool_)
        blocks = S.blocks[b,h] # (Lk_selected)
        for i, idx in enumerate(blocks.tolist()):
          if idx == -1: continue
          K_sel_b[i] = K[b, h, idx, :, :] 
          V_sel_b[i] = V[b, h, idx, :, :]

          #K_sel_b[i] = mx.where(padding_mask[idx, :], K_sel_b[1], mx.finfo(K.dtype).min)
          #V_sel_b[i] = mx.where(padding_mask[idx, :], V_sel_b[i], mx.finfo(K.dtype).min)

          start = idx*S.block_size
          stop = min(start + S.block_size, Lkv)
          k_pos = mx.arange(start, stop)
          q_pos = mx.arange(Lq).reshape(-1, 1)
          pk_start = i*(stop - start)
          pk_end = pk_start+(stop-start) 
          mask_b[:, pk_start:pk_end] = q_pos >= k_pos.reshape(1, -1)
        K_sel[b, h] = K_sel_b
        V_sel[b, h] = V_sel_b
        mask[b, h] = mask_b

    """
    for b in range(B):
      for h in range(Hkv):
        for blk in range(sel_max):
          for dim in range(S.block_size):
            print(f"\nBatch {b}, Head {h}, block {blk}, dim {dim}: ", end="")
            print(K_sel[b, h, blk, dim])
    """
    #print(K_sel, V_sel)
    #print(mask)
    K_sel = K_sel.reshape(B, Hkv, sel_max*S.block_size, D)
    V_sel = V_sel.reshape(B, Hkv, sel_max*S.block_size, D)
    return K_sel, V_sel, mask

    
  def __call__(self, x: mx.array, mask: mx.array, cache: Optional[tuple[mx.array, mx.array]]):
    start_g = time.perf_counter()
    self.cycles += 1
    B, Lq, _ = x.shape

    #start_t = time.perf_counter()
    Q = self.q_proj(x).reshape(B, Lq, self.n_heads, self.head_dim)
    Q = self.q_norm(Q).transpose(0, 2, 1, 3)

    if (self.cycles != 1 
        and self.cycles % self.S.update_frequency != 0
        and self.sparse_kv_cache[0].shape[2] >= self.kv_cache[0].shape[1] + self.cycles % self.S.update_frequency): # Quick decode using cached packed values

      # No strategy, packing or padding the full tensors 
      x = x[:, -1, :]
      K, V, padding_mask = self.sparse_kv_cache 
      Lkv = self.kv_cache[0].shape[1] + 1 

      new_V = self.v_proj(x).reshape(self.n_kv_heads, self.head_dim)
      V[-1, :, Lkv] = new_V
      self.kv_cache[1] = mx.concatenate([self.kv_cache[1], new_V.reshape(1, 1, self.n_kv_heads, self.head_dim)], axis=1)

      new_K = self.k_proj(x).reshape(B, -1, self.n_kv_heads, self.head_dim)
      new_K = self.k_norm(new_K).transpose(0, 2, 1, 3)
      new_K = self.rope(new_K, offset=Lkv) 
      new_K = new_K.reshape(self.n_kv_heads, self.head_dim)
      K[-1, :, Lkv, :] = new_K
      self.kv_cache[0] = mx.concatenate([self.kv_cache[0], new_K.reshape(1, 1, self.n_kv_heads, self.head_dim)], axis=1)
      Q = self.rope(Q, offset=Lkv) # offset rope

      # TODO: Add to normal cache too
      padding_mask[-1, Lkv, :, :] = False
      self.sparse_kv_cache = (K, V, padding_mask)

      #logger.info(f"[PROFILE] Quick decode: {(time.perf_counter() - start_t)*1000:0.5f}ms")

    else: 
      if self.cycles == 1: # Prefill
        #start_t = time.perf_counter()
        Lkv = Lq
        V = self.v_proj(x).reshape(B, -1, self.n_kv_heads, self.head_dim)
        K = self.k_proj(x).reshape(B, -1, self.n_kv_heads, self.head_dim)
        K = self.k_norm(K).transpose(0, 2, 1, 3)
        K = self.rope(K)
        Q = self.rope(Q)
        #print(Q.dtype, K.dtype, V.dtype)

        # Transpose back for S calculation
        K = K.transpose(0, 2, 1, 3)
        self.kv_cache = [K, V]
        #logger.info(f"[PROFILE] Prefill Projections: {(time.perf_counter() - start_t)*1000:0.5f}ms")

      else: # Full Decode: Recompute sparsity indices and masks 
        #start_t = time.perf_counter()
        K_cache, V_cache = self.kv_cache
        Q = self.rope(Q, offset=K_cache.shape[1]) # offset rope
        assert self.kv_cache is not None, "KV Cache is needed to compute decode phase."

        new_V = self.v_proj(x[:, -1, :]).reshape(B, -1, self.n_kv_heads, self.head_dim)
        V = mx.concatenate([V_cache, new_V], axis=1)

        new_K = self.k_proj(x[:, -1, :]).reshape(B, -1, self.n_kv_heads, self.head_dim)
        new_K = self.k_norm(new_K).transpose(0, 2, 1, 3)
        new_K = self.rope(new_K, offset=K_cache.shape[1]) 
        new_K = new_K.transpose(0, 2, 1, 3)
        K = mx.concatenate([K_cache, new_K], axis=1)
        #logger.info(f"[PROFILE] Long Decode Projections: {(time.perf_counter() - start_t)*1000:0.5f}ms")

      self.kv_cache = [K, V]

      Q = Q.transpose(0, 2, 1, 3)

      # Pad to block size
      #start_t = time.perf_counter()
      _, Lkv, Hkv, _ = K.shape
      final_lkv = math.ceil(Lkv / BLOCK_SIZE) * BLOCK_SIZE 
      padding = final_lkv - Lkv
      if padding > 0:
        K = mx.pad(K, [(0,0), (0,padding), (0,0), (0,0)], constant_values=0.0)
        V = mx.pad(V, [(0,0), (0,padding), (0,0), (0,0)], constant_values=0.0)
        padding_indices = mx.arange(final_lkv)[None, :, None, None]
        padding_mask = mx.where(padding_indices < Lkv, True, False)
      else:
        padding_mask = None
      #print(f"Strategy: {time.perf_counter() - start_t}s")
      #logger.info(f"[PROFILE] Padding {(time.perf_counter() - start_t)*1000:0.5f}ms")

      V = V.transpose(0, 2, 1, 3)

      #start_t = time.perf_counter()
      if not self.S:
        strat_in = StrategyInput(
          Q, K, mask, BLOCK_SIZE, gqa_interleave=False,
          last_q=None, gamma=0.1, min_budget=1, max_budget=128,
          tau=0.1, is_token_level=False)
        self.S = self.compute_strat(strat_in)
      elif self.cycles % self.S.update_frequency == 0:
          strat_in = StrategyInput(
            Q, K, mask, BLOCK_SIZE, gqa_interleave=False,
            last_q=None, gamma=0.1, min_budget=1, max_budget=128,
            tau=0.1, is_token_level=False)
          self.S = self.compute_strat.update(strat_in, prev_S=self.S, prev_Lkv=Lkv-1, cycles=self.cycles)
      #logger.info(f"[PROFILE] Strategy compute: {(time.perf_counter() - start_t)*1000:0.5f}ms")


      Q = Q.transpose(0, 2, 1, 3)
      K = K.transpose(0, 2, 1, 3)

      self.sparse_kv_cache = (K, V, padding_mask)

      # Packing the selected sparse blocks into dense buffers
      #start_t = time.perf_counter()
      if self.S.blocks.shape[2] != 1:
        K, V, sparse_mask = self.pack_kv_blocks(K, V, self.S, final_lkv, Lq, padding_mask=padding_mask)
        """
        if not isinstance(mask, str):
          mask += sparse_mask
        else:
          mask = sparse_mask 
        """
        mask = sparse_mask 
      #logger.info(f"[PROFILE] Packing and masking: {(time.perf_counter()-start_t)*1000:0.5f}ms")

    #print(Q.shape, K.shape, V.shape)
    #start_t = time.perf_counter()
    out = sparse_dot_product_attention_blocked(Q, K, V, self.scale, mask, padding_mask=padding_mask, selected_counts=None)
    #logger.info(f"[PROFILE] sdpa {(time.perf_counter() - start_t)*1000:0.5f}ms")
    out = self.o_proj(out)
    logger.info(f"[PROFILE] Global attention runtime: {(time.perf_counter() - start_g)*1000:0.5f}ms")

    return out
