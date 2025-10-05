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

Finally, generate protos:

```sh
uv run ./scripts/generate_protos.py
```

## Usage

dnet uses a **dynamic topology** approach where nodes start without models, then the API discovers devices and distributes layers optimally.

### Workflow

1. **Start Shards**: Launch shard nodes on each device (only need to specify ports)
2. **Start API**: Launch the API node
3. **Prepare Topology**: API discovers devices and solves for optimal layer distribution
4. **Load Model**: API instructs shards to load their assigned layers
5. **Inference**: Use `/v1/chat/completions` endpoint for generation

Supported models:

- Qwen3
- DeepSeek V2
- MLX formats: fp16, bf16, 4-bit, 8-bit quantized

### Running a Shard

Start a shard node with gRPC and HTTP ports:

```sh
uv run dnet-shard -p 6060 --http-port 7070
```

**Arguments:**

- `-p, --grpc-port`: gRPC server port (required)
- `--http-port`: HTTP server port (required)
- `-q, --queue-size`: Activation queue size (default: 10)
- `-w, --prefetch-window`: Number of layers to prefetch (default: 2)

The shard will:

- Register itself via mDNS discovery
- Wait for LoadModel commands from the API
- Process activations through assigned layers

### Running an API

Start the API node:

```sh
uv run dnet-api -p 8080
```

**Arguments:**

- `-p, --http-port`: HTTP server port (default: 8080)
- `-g, --grpc-port`: gRPC callback port (default: http-port + 1)

The API provides the following endpoints:

#### 1. Prepare Topology

Discover devices and compute optimal layer distribution:

```sh
curl -X POST http://localhost:8080/v1/prepare_topology \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "force_rediscover": false
  }'
```

**Response:**

```json
{
  "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  "num_layers": 24,
  "devices": [...],
  "assignments": [
    {"service_name": "shard-abc-hostname", "layers": [0, 1, 8, 9, 16, 17]},
    {"service_name": "shard-def-hostname", "layers": [2, 3, 10, 11, 18, 19]}
  ],
  "diagnostics": {...}
}
```

#### 2. Load Model

Load the model on shards with prepared topology:

```sh
curl -X POST http://localhost:8080/v1/load_model \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "assignments": [
      {"service_name": "shard-abc-hostname", "layers": [0, 1, 8, 9, 16, 17]},
      {"service_name": "shard-def-hostname", "layers": [2, 3, 10, 11, 18, 19]}
    ]
  }'
```

**Response:**

```json
{
  "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  "success": true,
  "shard_statuses": [
    {
      "service_name": "shard-abc-hostname",
      "success": true,
      "message": "Model loaded successfully",
      "layers_loaded": [0, 1, 8, 9, 16, 17],
      "load_time_ms": 1234.5
    }
  ],
  "total_load_time_ms": 2500.0
}
```

#### 3. Chat Completions

Generate text using the loaded model:

```sh
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
  }'
```

#### Other Endpoints

- `GET /health` - API health status
- `GET /devices` - List discovered devices
- `GET /ping` - Simple ping endpoint

> [!TIP]
>
> On MacOS, you can use `dns-sd` to check out devices over mDNS:
>
> ```sh
> dns-sd -Q _dnet_p2p._tcp.local. PTR
> ```
>
> See more instructions [here](https://github.com/firstbatchxyz/dnet-p2p?tab=readme-ov-file#discovering-with-dns-sd).

## Testing

You can lint the code using Ruff:

```sh
uvx ruff check
```

## License

You can find the license [here](./LICENSE).
