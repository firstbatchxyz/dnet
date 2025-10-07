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

**dnet** requires several submodules, which can all be cloned with the following command:

```sh
git clone --recurse-submodules https://github.com/firstbatchxyz/dnet.git
```

**dnet** uses `uv`, so make sure it is installed. You can check for uv with the command below, and follow the [installation guide](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it.

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

**dnet** uses a **dynamic topology** approach where nodes start without models, then the API discovers devices and distributes layers optimally.

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
uv run dnet-shard --http-port 8081 --grpc-port 58081
```

### Running an API

Start the API node:

```sh
uv run dnet-api --http-port 8080 --grpc-port 58080
```

To do inference, first, we must [prepare the topology](#prepare-topology) (discover nodes) and then [load the model](#load-model) itself.
After that, we can call the [completions](#chat-completions) endpoint as usual.

> [!TIP]
>
> We have a script that can prepare the model and load it at once:
>
> ```sh
> uv run ./scripts/prepare_model.py -m Qwen/Qwen3-4B-MLX-4bit
> ```

#### Prepare Topology

Discover devices and compute optimal layer distribution:

```sh
curl -X POST http://localhost:8080/v1/prepare_topology \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-MLX-4bit"
  }'
```

Response will be in the form:

```json
{
  "model": "Qwen/Qwen3-4B-MLX-4bit",
  "num_layers": 36,
  "devices": [
    {
      "service_name": "shard-123456._dnet_p2p._tcp.local.",
      "local_ip": "192.168.1.2",
      "http_port": 8081,
      "grpc_port": 58081
    }
  ],
  "assignments": [
    {
      "service": "shard-123456._dnet_p2p._tcp.local.",
      "layers": [
        [0, 1, 2],
        [3, 4, 5]
      ],
      "next_services": ["shard-123456._dnet_p2p._tcp.local.", null]
    }
  ]
}
```

#### Load Model

Load the model on shards with prepared topology:

<!-- add devices to body here as well -->

```sh
curl -X POST http://localhost:8080/v1/load_model \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-MLX-4bit",
    "assignments": [
      {
        "service": "shard-123456._dnet_p2p._tcp.local.",
        "layers": [[0, 1, 2], [3, 4, 5]],
        "next_services": ["shard-987654._dnet_p2p._tcp.local.", null]
      },
      {
        "service": "shard-987654._dnet_p2p._tcp.local.",
        "layers": [[6, 7, 8], [9, 10, 11]],
        "next_services": ["shard-123456._dnet_p2p._tcp.local.", null]
      }
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

#### Chat Completions

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

#### Devices

You can get the list of discoverable devices with:

```sh
curl http://localhost:8080/v1/devices \
  -H "Content-Type: application/json"
```

> [!TIP]
>
> You can use `dns-sd` to check out devices over mDNS:
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
