

#include <stdio.h>
#include <atomic>
#include <vector>
#include <mutex> 
#include <iostream>
#include <stdexcept>

#include <simd/simd.h>
#include <Metal/metal.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Paged and Sparse KV Cache

constexpr size_t BLOCK_SIZE = 128; // tokens 
constexpr size_t MAX_BLOCKS = 2 << 14; 

// Lock-free Treiber list implementation
// From : https://people.csail.mit.edu/shanir/publications/Lock_Free.pdf
struct Node {
  int block_id = -1;
  std::atomic<Node*> next{nullptr};
};

struct KVBlock {
  int physical_idx = -1;
};

// Use large single buffers for performance
class PageAllocator {
  private:
    MTL::Device* m_device;
    MTL::Buffer* k_cache;
    MTL::Buffer* v_cache;
    std::vector<Node*> node_pool;
    std::atomic<int> free_node_idx{-1};
    std::atomic<Node*> HEAD{nullptr};
    size_t bytesize = 0;
    size_t nr_kv_heads = 0;
    size_t block_size = 0;
    size_t head_dim = 0;

  public:
    PageAllocator(size_t nr_heads, size_t kv_bytes, size_t head_dim) 
      : nr_kv_heads(nr_heads), head_dim(head_dim),
        bytesize(kv_bytes) 
    {
      m_device = MTL::CreateSystemDefaultDevice(); 
      if(!m_device) {
        throw std::runtime_error("Failed to create Metal device.");
      }
      k_cache = m_device->newBuffer(MAX_BLOCKS*bytesize, MTL::ResourceStorageModeShared);
      v_cache = m_device->newBuffer(MAX_BLOCKS*bytesize, MTL::ResourceStorageModeShared);

      // node pool
      node_pool.reserve(nr_heads * MAX_BLOCKS);
      for(int i=0; i< nr_heads*MAX_BLOCKS; ++i) {
        node_pool.emplace_back(new Node());
      }
      free_node_idx.store(nr_heads*MAX_BLOCKS - 1);

      HEAD.store(nullptr, std::memory_order_relaxed);
      for(int i=0; i< nr_heads*MAX_BLOCKS; ++i) {
        push(i);
      }
    }

    ~PageAllocator() {
      k_cache->release();
      v_cache->release();
      for(auto* node: node_pool) {
          delete node;
      }
    }

    int allocate_block() {
      Node* old_head;
      do {
        old_head = HEAD.load(std::memory_order_acquire);
        if(!old_head) {
          throw std::runtime_error("No free blocks");
        }
      } while(!HEAD.compare_exchange_weak(old_head, 
              old_head->next.load(std::memory_order_relaxed), 
              std::memory_order_release));
      int block = old_head->block_id;
      recycle_node(old_head);
      return block;
    }

    int free_block(int block) {
      Node* node = get_node_from_pool();
      if(!node) {
        throw std::runtime_error("Node pool exhausted");
      }
      node->block_id = block;
      Node* old_head;
      do {
        old_head = HEAD.load(std::memory_order_acquire);
        node->next.store(old_head, std::memory_order_relaxed);
      } while(!HEAD.compare_exchange_weak(old_head, node, std::memory_order_release));
    }

    MTL::Device* device() { return m_device; }
    MTL::Buffer* get_k_cache() { return k_cache; }
    MTL::Buffer* get_v_cache() { return v_cache; }

    // Debug helpers
    std::vector<simd_float16> get_block(bool key, size_t b, size_t h, size_t bid) {
      //std::lock_guard<std::mutex> lock(mux);
      MTL::Buffer* cache = key ? k_cache : v_cache; 
      size_t offset = (b*nr_kv_heads*MAX_BLOCKS + h*MAX_BLOCKS + bid);
      simd_float16* data = reinterpret_cast<simd_float16*>(cache->contents()) + offset;
      return std::vector<simd_float16>(data, data+block_size);
    }

  private:
    Node* get_node_from_pool() {
      int idx = free_node_idx.fetch_sub(1, std::memory_order_relaxed);
      if(idx<0) {
        free_node_idx.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
      }
      return node_pool[idx];
    }

    void recycle_node(Node* node) {
      int idx = free_node_idx.fetch_add(1, std::memory_order_relaxed) + 1;      
      if(idx > (int)MAX_BLOCKS) {
        std::cerr << "Node pool overflow." << std::endl;
        return;
      }
      node_pool[idx] = node;
      node->next.store(nullptr, std::memory_order_relaxed);
    }

    // Grab new block and move HEAD
    void push(int block) {
      Node* n_node = get_node_from_pool();
      n_node->block_id = block;
      Node* old_head;
      do {
        old_head = HEAD.load(std::memory_order_acquire);
        n_node->next.store(old_head, std::memory_order_relaxed);
      } while(!HEAD.compare_exchange_weak(old_head, n_node, 
                                          std::memory_order_release, 
                                          std::memory_order_relaxed));
    }
};

class SparseKVCache {
  private:
    std::mutex mux;
    PageAllocator* alloc;
    MTL::CommandQueue* queue;
    std::vector<std::vector<std::vector<KVBlock>>> page_table; // [B, head, block_id]
    MTL::Buffer* page_table_buffer = nullptr;
    size_t bytesize = 0; // item byte size (default f16 - 4)
    std::vector<size_t> seq_len; // Sequence lengths of each batch 
    std::vector<size_t> sq_offs; // Sequence offsets 
    size_t batch_size = 0;
    size_t block_size = 0;
    size_t num_kv_heads = 0;
    size_t head_dim = 0;
    size_t max_blocks = MAX_BLOCKS;

  public:
    SparseKVCache(PageAllocator* allocator, size_t bytes, size_t block_size,
                  size_t batch_size, size_t head_size, size_t num_heads)
                       : alloc(allocator), batch_size(batch_size), block_size(block_size),
                         head_dim(head_size), num_kv_heads(num_heads), 
                         seq_len(batch_size, 0), bytesize(bytes), 
                         page_table(batch_size, std::vector<std::vector<KVBlock>>(num_heads)) 
    {
      queue = alloc->device()->newCommandQueue();
      seq_len.reserve(batch_size);
      sq_offs.reserve(batch_size);

      for(int b=0; b<batch_size; b++) {
        seq_len[b] = -1;
        sq_offs[b] = -1;
      }
    }

    void allocate_blocks(size_t b, size_t h, size_t blocks) {
      auto& pt = page_table[b][h];
      while(pt.size() <= blocks) {
        KVBlock n_blk;
        n_blk.physical_idx = alloc->allocate_block();
        if(n_blk.physical_idx == -1) {
          throw std::runtime_error("Failed to allocate new physical block.");
        }
        pt.push_back(n_blk);
      }
    }

    // Non-blocking append function
    void append(MTL::Buffer* new_k, 
                MTL::Buffer* new_v, 
                std::vector<size_t> num_new_tokens)
    {
      std::lock_guard<std::mutex> lock(mux);

      if(num_new_tokens.size() != batch_size) {
        throw std::runtime_error("New token counts don't match known batch size.");
      }

      MTL::CommandBuffer* cmd_buf = queue->commandBuffer();
      MTL::BlitCommandEncoder* blit = cmd_buf->blitCommandEncoder();

      // new_k, new_v of shape [B, 1, Hkv, D]
      for (size_t b=0; b<batch_size; ++b) {
        if (seq_len[b] == 0) continue;
        if(sq_offs[b] == -1) {
          for(int i=0; i<b; i++) { // Cumulative batch seqlen sum
            sq_offs[b] += seq_len[i] * num_kv_heads * head_dim * sizeof(simd_float16); 
          }
          sq_offs[b] += seq_len[b] * num_kv_heads * head_dim * sizeof(simd_float16);
        }

        for(int tok=0; tok<num_new_tokens[b]; tok++) {
          size_t src_off = b*tok*num_kv_heads*head_dim*sizeof(simd_float16);
          size_t dst_off = sq_offs[b];
          size_t cpy_bytes = num_kv_heads*head_dim * sizeof(simd_float16);

          // Bulk copy of values 
          blit->copyFromBuffer(new_k, src_off, alloc->get_k_cache(), dst_off, cpy_bytes);
          blit->copyFromBuffer(new_v, src_off, alloc->get_v_cache(), dst_off, cpy_bytes);

          // Set block metadata
          size_t current_tok = seq_len[b];
          size_t remaining = num_new_tokens[b];
          size_t pos = 0;
          while(remaining > 0) {
            size_t num_blocks = current_tok / BLOCK_SIZE;
            size_t block_offset = current_tok % BLOCK_SIZE;
            size_t to_copy = std::min(remaining, BLOCK_SIZE - block_offset);
            for(size_t h=0; h<num_kv_heads; h++) {
              allocate_blocks(b, h, num_blocks);
            }
            pos += to_copy;
            remaining -= to_copy;
            current_tok += to_copy;
          }
        }
        seq_len[b] += num_new_tokens[b];
        sq_offs[b] += num_new_tokens[b] * num_kv_heads*head_dim*sizeof(simd_float16); 
      }
      blit->endEncoding();
      cmd_buf->commit();
    }
    
    // Pack working physical block indices according to sparsity mask
    void pack_table_buffer(MTL::Buffer* page_table_buff, MTL::Buffer* mask) {
      std::lock_guard<std::mutex> lock(mux);

      size_t max_blocks = 0;
      for(size_t b=0; b<batch_size; b++) {
        for(size_t h=0; h<num_kv_heads; h++) {
          max_blocks = std::max(max_blocks, page_table[b][h].size());
        }
      }
    
      if(mask->length() < batch_size*num_kv_heads*max_blocks*sizeof(unsigned char)) {
        throw std::runtime_error("Sparsity mask is too small for KV Cache.");
      }

      unsigned char* mask_data = reinterpret_cast<unsigned char*>(mask->contents());

      // Temp buffer that stores all indices + offset and count metadata
      // of blocks that are active in the sparse_mask

      // [total_blocks * sizeof(int)] indices of active physical blocks 
      // [batch_size * num_kv_heads * sizeof(unsigned int)] offsets (starting point of indices array)
      // [batch_size * num_kv_heads * sizeof(unsigned int)] counts (number of active blocks for b and h)
      size_t max_buffer_size = batch_size*num_kv_heads*max_blocks*sizeof(int) + 2*batch_size*num_kv_heads*sizeof(unsigned int);
      MTL::Buffer* temp = alloc->device()->newBuffer(max_buffer_size, MTL::ResourceStorageModeShared);
      int* temp_idx = reinterpret_cast<int*>(temp->contents());
      unsigned int* temp_off = reinterpret_cast<unsigned int*>(temp_idx + batch_size * num_kv_heads * max_blocks);
      unsigned int* temp_counts = temp_off + batch_size*num_kv_heads; 

      size_t total_blocks = 0;
      std::vector<size_t> block_counts(batch_size*num_kv_heads, 0);
      for(size_t b=0; b<batch_size; b++) {
        for(size_t h=0; h<num_kv_heads; h++) {
          size_t bh_idx = b*num_kv_heads + h;
          temp_off[bh_idx] = total_blocks;
          for(size_t i=0; i<max_blocks; i++) {
            size_t mask_idx = (b*num_kv_heads + h) * max_blocks + i;
            if(mask_data[mask_idx] && page_table[b][h][i].physical_idx != -1) {
              temp_idx[total_blocks++] = page_table[b][h][i].physical_idx;
              block_counts[bh_idx]++;
            }
          }
          temp_counts[bh_idx] = block_counts[bh_idx];
        }
      }
      
      // Resize
      size_t real_buffer_size = total_blocks * sizeof(int) + 2*batch_size*num_kv_heads*sizeof(unsigned int);
      if(page_table_buff && page_table_buff->length() != real_buffer_size) {
        page_table_buff->release();
        page_table_buff = nullptr;
      }
      if(!page_table_buff) {
        page_table_buff = alloc->device()->newBuffer(real_buffer_size, MTL::ResourceStorageModeShared);
      }

      // Copy to final buffer
      MTL::CommandBuffer* cmd_buf = queue->commandBuffer();
      MTL::BlitCommandEncoder* blit = cmd_buf->blitCommandEncoder();
      blit->copyFromBuffer(temp, 0, page_table_buff, 0, real_buffer_size);
      blit->endEncoding();
      cmd_buf->commit();
      cmd_buf->waitUntilCompleted();
      temp->release();

      page_table_buffer = page_table_buff;
    }

    std::pair<MTL::Buffer*, MTL::Buffer*>
    read(MTL::Buffer* sparse_mask) {
      std::lock_guard<std::mutex> lock(mux);

      MTL::Buffer* sparse_block_idxs;
      pack_table_buffer(sparse_block_idxs, sparse_mask);
      int* temp_indices = reinterpret_cast<int*>(sparse_block_idxs->contents());
      unsigned int* temp_off = reinterpret_cast<unsigned int*>(temp_indices + batch_size*num_kv_heads*max_blocks);
      unsigned int* temp_counts = temp_off + batch_size*num_kv_heads;

      // Get output sizes
      size_t max_tokens = 0;
      std::vector<size_t> b_sizes(batch_size, 0);
      for(size_t b=0; b<batch_size; b++) {
        size_t total_tokens = 0;
        for(size_t h=0; h<num_kv_heads; h++) {
          size_t bh_idx = b*num_kv_heads+h;
          size_t block_count = temp_counts[bh_idx];
          for(size_t i=0; i<block_count; i++) {
            total_tokens += std::min(block_size, seq_len[b] - i * block_size);
          }
        }
        b_sizes[b] = total_tokens * head_dim * num_kv_heads;
        max_tokens = std::max(max_tokens, total_tokens);
      }
      size_t max_size = max_tokens * head_dim * num_kv_heads * bytesize;

      // Allocate buffers
      MTL::Buffer* k_out = alloc->device()->newBuffer(max_size, MTL::ResourceStorageModeShared);
      MTL::Buffer* v_out = alloc->device()->newBuffer(max_size, MTL::ResourceStorageModeShared);

      MTL::CommandBuffer* cmd_buf = queue->commandBuffer();
      MTL::BlitCommandEncoder* blit = cmd_buf->blitCommandEncoder();

      // Gather values from indices 
      for(size_t b=0; b<batch_size; b++) {
        size_t out_pos = 0;
        for(size_t h=0; h<num_kv_heads; h++) {
          size_t bh_idx = b*num_kv_heads+h;
          for(size_t i=0; i<temp_counts[bh_idx]; i++) {
            int physical_block = temp_indices[temp_off[bh_idx] + i];
            size_t tokens_in_block = std::min(block_size, seq_len[b] - i*block_size);
            size_t src_offset = physical_block * block_size * head_dim * bytesize;
            size_t dst_offset = out_pos * head_dim * num_kv_heads + h * head_dim * bytesize;
            size_t copy_size = tokens_in_block * head_dim * bytesize;

            blit->copyFromBuffer(alloc->get_k_cache(), src_offset, k_out, dst_offset, copy_size);
            blit->copyFromBuffer(alloc->get_v_cache(), src_offset, v_out, dst_offset, copy_size);
            out_pos += tokens_in_block;
          }
        }
      }
      blit->endEncoding();
      cmd_buf->commit();
      cmd_buf->waitUntilCompleted();
      sparse_block_idxs->release();
      return {k_out, v_out};
    }

    // LRU block eviction from sparse mask history
    void evict_block(size_t b, size_t h, size_t block) {}

    // Debug helpers
    std::vector<size_t>* get_seq_offsets() { return &sq_offs; }
    std::vector<size_t>* get_seq_len() { return &seq_len; }

    std::vector<std::vector<std::vector<KVBlock>>> get_page_table() {
      std::vector<std::vector<std::vector<KVBlock>>> ret(batch_size, std::vector<std::vector<KVBlock>>(num_kv_heads));
      for(size_t b=0; b<batch_size; b++){
        for(size_t h=0; h<num_kv_heads; h++){
          ret[b][h].reserve(page_table[b][h].size());
          for(const auto& i : page_table[b][h]) {
            ret[b][h].push_back(i);
          }
        }
      }
      return ret;
    }

    std::vector<simd_float16> get_data(bool key, size_t b, size_t h, size_t block) {
      if(b >= batch_size || h >= num_kv_heads || block >= MAX_BLOCKS) {
        throw std::runtime_error("Invalid request for 'get_block': {b}, {h}, {block}");
      }
      return alloc->get_block(key, b, h, block);
    }
};


// Bind to python
PYBIND11_MODULE(kv_cache, m) {
  py::class_<PageAllocator, std::shared_ptr<PageAllocator>>(m, "PageAllocator")
    .def(py::init<size_t, size_t, size_t>());

  py::class_<SparseKVCache, std::shared_ptr<SparseKVCache>>(m, "SparseKVCache")
    .def(py::init<PageAllocator*, size_t, size_t, size_t, size_t, size_t>())
    .def("append", &SparseKVCache::append)
    .def("pack_table_buffer", &SparseKVCache::pack_table_buffer)
    .def("read", &SparseKVCache::read)
    .def("get_seq_len", &SparseKVCache::get_seq_len)
    .def("get_seq_offset", &SparseKVCache::get_seq_offsets)
    .def("get_page_table", &SparseKVCache::get_page_table)
    .def("data", &SparseKVCache::get_data);
}
