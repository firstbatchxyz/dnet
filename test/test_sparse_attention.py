

import mlx.core as mx
import mlx.nn as nn
import pytest
import math

from src.runtime.sparse_attention import FlexPrefillSparseAttention, SparseAttention, StrategyInput, BLOCK_SIZE

#@pytest.fixture
def fixed_inputs():
  B = 1
  Lq = BLOCK_SIZE*2
  Lkv = BLOCK_SIZE*4
  Hq = 32
  Hkv = 16 # GQA n_repeats=2
  D = 64 
  return B, Lq, Lkv, Hq, Hkv, D

# Test deterministic arange values
#@pytest.mark.parametrize("gamma", [0.3, 0.5, 0.8])
#@pytest.mark.parametrize("Lk", [BLOCK_SIZE*8, BLOCK_SIZE*16])
def test_query_aware_search(fixed_inputs, gamma, Lk):
  B, Lq, Lkv, Hq, Hkv, D = fixed_inputs
  Lkv = Lk
  Q = mx.arange(B*Lq*Hq*D, dtype=mx.float32).reshape(B, Lq, Hq, D)
  K = mx.arange(B*Lkv*Hkv*D, dtype=mx.float32).reshape(B, Lkv, Hkv, D)
  strat = FlexPrefillSparseAttention()
  input = StrategyInput(
    Q=Q,
    K=K,
    block_size=BLOCK_SIZE,
    gamma=gamma,
    min_budget=1,
    max_budget=128,
    tau=0.5,
    mask=None,
  )
  blocks = strat.query_aware_search(Q, K, input.gamma, D, input.min_budget, input.max_budget)
  num_valid = mx.sum(blocks >= 0, axis=-1)
  assert blocks.shape == (B, Hkv, input.max_budget), "Shape mismatch"
  assert mx.all(num_valid >= input.min_budget) & mx.all(num_valid < input.max_budget), "Invalid number of blocks selected"

  # Per head checks
  num_blocks = math.ceil(K.shape[1] / BLOCK_SIZE)
  for b in range(B):
    for h in range(Hkv):
      #print(num_valid[b, h])
      valid = blocks[b, h, :int(num_valid[b, h].item())]
      assert valid.size == num_valid[b, h]
      assert mx.all(valid >= 0) and mx.all(num_blocks > valid)
      #assert mx.all( ~(blocks[b,h, 1:] != blocks[b, h, :1])), "Repeting indices"
      #print( ~(blocks[b,h, 1:] != blocks[b, h, :1]))
  
def test_query_aware_search_variable_gamma(fixed_inputs):
  B, Lq, Lkv, Hq, Hkv, D = fixed_inputs
  Q = mx.random.normal([B, Lq, Hq, D], scale=2.0)
  K = mx.random.normal([B, Lkv, Hkv, D], scale=2.0)
  strat = FlexPrefillSparseAttention()
  input = StrategyInput(
    Q=Q,
    K=K,
    block_size=BLOCK_SIZE,
    gamma=0.3,
    min_budget=1,
    max_budget=128,
    tau=0.5,
    mask=None,
  )
  blocks = strat.query_aware_search(Q, K, input.gamma, D, input.min_budget, input.max_budget)
  num_valid = mx.sum(blocks >= 0, axis=-1)
  num_blocks = math.ceil(K.shape[1] / BLOCK_SIZE)
  for b in range(B):
    for h in range(Hkv):
      valid = blocks[b, h, :int(num_valid[b, h].item())]
      assert valid.size == num_valid[b, h]
      assert mx.all(valid >= 0) and mx.all(num_blocks > valid)
      #assert mx.all( ~(blocks[b,h, 1:] != blocks[b, h, :1])), "Repeting indices"

def test_vertical_slash_search(fixed_inputs):
  B, Lq, Lkv, Hq, Hkv, D = fixed_inputs
  Q = mx.arange(B*Lq*Hq*D, dtype=mx.float32).reshape(B, Lq, Hq, D)
  K = mx.arange(B*Lkv*Hkv*D, dtype=mx.float32).reshape(B, Lkv, Hkv, D)
  strat = FlexPrefillSparseAttention()
  input = StrategyInput(
    Q=Q,
    K=K,
    block_size=BLOCK_SIZE,
    gamma=0.5,
    min_budget=1,
    max_budget=64,
    tau=0.5,
    mask=None,
  )
  blocks = strat.vertical_slash_search(Q, K, input.gamma, D, input.min_budget, input.max_budget)

def test_vertical_slash_search_normal_dist(fixed_inputs):
  B, Lq, Lkv, Hq, Hkv, D = fixed_inputs
  Q = mx.random.normal([B, Lq, Hq, D], scale=2.0)
  K = mx.random.normal([B, Lkv, Hkv, D], scale=2.0)
  strat = FlexPrefillSparseAttention()
  input = StrategyInput(
    Q=Q,
    K=K,
    block_size=BLOCK_SIZE,
    gamma=0.5,
    min_budget=1,
    max_budget=64,
    tau=0.5,
    mask=None,
  )
  blocks = strat.vertical_slash_search(Q, K, input.gamma, D, input.min_budget, input.max_budget)
  print(blocks)
  

if __name__ == "__main__":
  input  = fixed_inputs()
  test_query_aware_search(input, 0.5, BLOCK_SIZE*8)
  test_query_aware_search_variable_gamma(input)
  test_vertical_slash_search(input)
  test_vertical_slash_search_normal_dist(input)
