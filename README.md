<p align="center">
  <img src="https://raw.githubusercontent.com/firstbatchxyz/.github/refs/heads/master/branding/dria-logo-square.svg" alt="logo" width="168">
</p>

<p align="center">
  <h1 align="center">
    dnet
  </h1>
  <p align="center">
    <i>Distributed LLM inference on Apple Silicon using ring topology.</i>
  </p>
</p>

<p align="center">
    <a href="https://opensource.org/license/apache-2-0" target="_blank">
        <img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache%202.0-7CB9E8.svg">
    </a>
</p>

## Installation

dnet requires several submodules, which can all be cloned with the following command:

```sh
git clone --recurse-submodules https://github.com/firstbatchxyz/dnet.git
```

dnet uses `uv`, so make sure it is installed. You can check for uv with the command below, and follow the [installation guide](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it.

```sh
uv --version
```

After cloning the repository, simply run the following to setup everything:

```sh
uv sync
```

## Setup

To use dnet, start multiple [shards](#running-a-shard) and a single [API](#running-an-api).

### Running an API

### Running a Shard

## Testing

You can lint the code using Ruff:

```sh
uvx ruff check
```

---

---

---

---

---

## Quick Start

Run a model across 2 local shards:

```bash
# Terminal 1: Start first shard (layers 0-17)
uv run piped_mlx_ring_shard \
  -m "Qwen/Qwen3-4B-MLX-8bit" \
  -p 50501 \
  -l "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]" \
  -n "localhost:50502" \
  --prefetch-window 18

# Terminal 2: Start second shard (layers 18-35)
uv run piped_mlx_ring_shard \
  -m "Qwen/Qwen3-4B-MLX-8bit" \
  -p 50502 \
  -l "[18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]" \
  -n "localhost:8080" \
  --prefetch-window 18

# Terminal 3: Start API server
uv run piped_mlx_ring_api \
  -a "localhost:8080" \
  -s "localhost:50501" \
  -m "Qwen/Qwen3-4B-MLX-8bit"
```

> [!TIP]
>
> On MacOS, you can use `dns-sd` to check out devices over mDNS:
>
> ```sh
> dns-sd -Q _dnet_p2p._tcp.local. PTR
> ```
>
> See more instructions [here](https://github.com/firstbatchxyz/dnet-p2p?tab=readme-ov-file#discovering-with-dns-sd).

### Test

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default_model",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 30
  }'
```

## Multi-Machine Setup

**Machine 1:**

```bash
# Shard 1
uv run piped_mlx_ring_shard \
  -m "Qwen/Qwen3-4B-MLX-8bit" \
  -p 50501 \
  -l "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]" \
  -n "192.168.1.102:50502" \
  --prefetch-window 18

# API Server (separate terminal)
uv run piped_mlx_ring_api \
  -a "0.0.0.0:8080" \
  -s "192.168.1.101:50501" \
  -m "Qwen/Qwen3-4B-MLX-8bit"
```

**Machine 2:**

```bash
# Shard 2
uv run piped_mlx_ring_shard \
  -m "Qwen/Qwen3-4B-MLX-8bit" \
  -p 50502 \
  -l "[18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]" \
  -n "192.168.1.101:8080" \
  --prefetch-window 18
```

## Multi-Round Setup (Memory Efficient)

For very large models that don't fit in memory, use windowed prefetching. Each device processes multiple layer groups in rounds, loading only `w` layers at a time.

### Example: Qwen3-32B across 2 devices (64 layers total)

**Device 1:**

```bash
# Handles layers 0-17 and 36-53 with window size 18
# Round 1: Load and process layers 0-17
# Round 3: Load and process layers 36-53
uv run piped_mlx_ring_shard \
  -m "Qwen/Qwen3-32B-MLX-bf16" \
  -p 50501 \
  -l "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]" \
  -n "192.168.1.102:50502" \
  --prefetch-window 18

# API Server (separate terminal)
uv run piped_mlx_ring_api \
  -a "0.0.0.0:8080" \
  -s "192.168.1.101:50501" \
  -m "Qwen/Qwen3-32B-MLX-bf16"
```

**Device 2:**

```bash
# Handles layers 18-35 and 54-63 with window size 18
# Round 2: Load and process layers 18-35
# Round 4: Load and process layers 54-63
uv run piped_mlx_ring_shard \
  -m "Qwen/Qwen3-32B-MLX-bf16" \
  -p 50502 \
  -l "[18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,54,55,56,57,58,59,60,61,62,63]" \
  -n "192.168.1.101:8080" \
  --prefetch-window 18
```

### How Multi-Round Works

With `--prefetch-window 18` and 36/28 layers split across 2 devices:

**Device 1 (36 layers total, 18 at a time):**

1. **Round 1**: Load layers 0-17 into memory, process token, pass to Device 2
2. **Round 3**: Load layers 36-53 into memory, process token, pass to Device 2

**Device 2 (28 layers total, 18 at a time):**

1. **Round 2**: Load layers 18-35 into memory, process token, pass to Device 1
2. **Round 4**: Load layers 54-63 into memory, process token, return to API

Each device only keeps 18 layers in memory (~9GB for 32B model), allowing you to run the full 32B model with limited RAM per device. The ring topology ensures activations flow through all 64 layers in the correct sequence.

## Commands

### Shard Server

```bash
uv run piped_mlx_ring_shard -m MODEL -p PORT -l LAYERS -n NEXT_SHARD [OPTIONS]
```

- `-m`: Hugging Face model ID
- `-p`: gRPC port
- `-l`: Layer indices (e.g., "[0,1,2,3]")
- `-n`: Next shard address
- `--prefetch-window`: Layers to prefetch (default: layer count)

### API Server

```bash
uv run piped_mlx_ring_api -a API_ADDR -s SHARD_ADDR -m MODEL
```

- `-a`: HTTP API address
- `-s`: First shard's gRPC address
- `-m`: Model ID (must match shards)

## Supported Models

- Qwen3
- DeepSeek V2
- MLX formats: fp16, bf16, 4-bit, 8-bit quantized

## Utilities

```bash
# Kill all processes
pkill -f piped_mlx_ring

# Check ports
lsof -i :8080,50501,50502

# Development setup
make dev

# Clean build artifacts
make clean
```
