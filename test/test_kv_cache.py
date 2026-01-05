

import os
import sys
import ctypes
import pathlib
import mlx.nn as nn
import mlx.core as mx

import pathlib
import traceback
import importlib, importlib.machinery, importlib.util
#so = pathlib.Path("../lib/kv_cache/").resolve().glob("kv_cache.dylib")
so = pathlib.Path(__file__).resolve().parents[1] / "lib" / "kv_cache" / "kv_cache.dylib"
ldr = importlib.machinery.ExtensionFileLoader("kv_cache", str(so))
spec = importlib.util.spec_from_loader("kv_cache", ldr)
kv_cache = importlib.util.module_from_spec(spec)

try:
  ldr.exec_module(kv_cache)
except Exception:
  traceback.print_exec()

import pytest

@pytest.fixture
def kv_cache_setup():
  num_kv_heads = 8
  block_size = 128
  max_blocks = 256 
  batch_size = 1
  head_dim = 128
  bytesize = 2

  allocator = kv_cache.PageAllocator(num_kv_heads, block_size, head_dim)
  cache = kv_cache.SparseKVCache(allocator, bytesize, batch_size, block_size, head_dim, num_kv_heads)

  yield cache, num_kv_heads, block_size, head_dim, batch_size, max_blocks

# Append data to cache
@pytest.fixture(params=[1, 284, 8491])
def appended_cache(kv_cache_setup, request):
  cache, num_kv_heads, block_size, head_dim, batch_size, _ = kv_cache_setup
  num_new_tokens = request.param
  k_data = mx.random.normal([1, num_new_tokens, num_kv_heads, head_dim])
  v_data = mx.random.normal([1, num_new_tokens, num_kv_heads, head_dim])
  cache.append(k_data, v_data, 1)
  return cache, num_new_tokens 

# Test sequence size and offsets after appending
def test_append(kv_cache_setup, appended_cache):
  _, num_kv_heads, _, _, batch_size, _ = kv_cache_setup
  cache, num_new_tokens = appended_cache
  seq_len = cache.get_seq_len()
  seq_offset = cache.get_seq_offset()
  for b in range(batch_size):
    assert seq_len[b] == num_new_tokens, f"Batch {b} seq_len: {cache.seq_len[b]} != {num_new_tokens}" 
    assert seq_offset[b] == num_new_tokens * num_kv_heads, \
      f"Batch {b} seq_offset: {seq_offset[b]} != {num_new_tokens*num_kv_heads}"

# Ensure page table is sane
def test_page_table(kv_cache_setup, appended_cache):
  _, num_kv_heads, block_size, _, batch_size, _ = kv_cache_setup
  cache, num_new_tokens = appended_cache 
  page_table = cache.get_page_table()
  for b in range(batch_size):
    num_blocks = (num_new_tokens + block_size - 1) // block_size
    for h in range(num_kv_heads):
      assert len(page_table[b][h] == num_blocks), \
        f"Batch {b} Head {h}: page table len mispatch: {len(page_table[b][h])} : {num_blocks}"
      for i in range(num_blocks):
        assert page_table[b][h][i] >=0, f"Batch {b}, Head {h}, Block{i} is invalid"

# Test packing from sparsity mask
def test_pack_table_buffer():
  pass

def test_read(kv_cache_setup, appended_cache):
  pass



