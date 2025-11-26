<p align="center">
  <img src="https://raw.githubusercontent.com/firstbatchxyz/.github/refs/heads/master/branding/dria-logo-square.svg" alt="logo" width="168">
</p>

<p align="center">
  <h1 align="center">
    dnet
  </h1>
  <p align="center">
    <i>Distributed LLM Inference for Apple Silicon Clusters</i>
  </p>
</p>

<p align="center">
    <a href="https://opensource.org/license/apache-2-0" target="_blank">
        <img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache%202.0-D22128.svg?logo=apache">
    </a>
    <a href="./.github/workflows/ci.yml" target="_blank">
        <img alt="Workflow: Tests" src="https://github.com/firstbatchxyz/dnet/actions/workflows/ci.yml/badge.svg?branch=master">
    </a>
    <a href="https://discord.com/invite/XwxZFkpFuQ" target="_blank">
        <img alt="License: Apache-2.0" src="https://img.shields.io/badge/discord-Dria-5865F2.svg?logo=discord">
    </a>

</p>

**RUN BIG MODELS | RUN LONG CONTEXT | MAXIMIZE UTILIZATION**

**dnet** runs LLMs across Apple Silicon devices. Modular execution strategies, automatic device profiling, drop-in OpenAI API.


## Features

- **Execution**
  - **No Memory Ceiling**: Run models that exceed total cluster memory—compute/I/O overlap keeps data flowing
  - **UMA specific**: Designed for Apple Silicon's unified memory for efficient layer swapping
  - **OpenAI-Compatible**: Drop-in `/v1/chat/completions` endpoint

- **Cluster Management**
  - **Automatic Discovery**: Nodes find each other; no manual topology configuration
  - **Thunderbolt Detection**: Automatically utilizes Thunderbolt for high-bandwidth inter-device communication

- **Workload Assignment**
  - **Device Profiling**: Measures FLOPs, memory, and inter-device latency per node
  - **Model Profiling**: Analyzes compute and memory requirements per layer
  - **Heterogeneity-Aware Solver**: Topology aware assignment that accounts for device capability, network speed, KV cache size, and disk speed

- ✅ **[Pipelined-ring](https://arxiv.org/pdf/2504.08791)** – Run >32B 8-bit models across devices with insufficient total memory
- 🚧 **Long context** – Make >128K context windows a reality for home clusters
- 🚧 **High throughput** – Maximize throughput via tensor parallelism
- 🚧 **Unified backend** – A single optimized backend for Apple Silicon, NVIDIA, and AMD (currently Apple Silicon only, via MLX)

## Installation

**dnet** requires several submodules, which can all be cloned with the following command:

```sh
git clone --recurse-submodules https://github.com/firstbatchxyz/dnet.git
```

**dnet** uses `uv`, so make sure it is installed. You can check for uv with the command below, and follow the [installation guide](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it.

```sh
uv --version
```

**dnet** currently only supports MLX on Apple Silicon. To install, run:

```sh
uv sync --extra mac
```

After syncing dependencies, generate protos:

```sh
uv run ./scripts/generate_protos.py
```
**dnet** supports make commands, run `make protos` to generate protos.

## Usage

**dnet** uses a **dynamic topology** approach where nodes start without models, then the API discovers devices and distributes layers optimally using [distilp](https://github.com/firstbatchxyz/distilp).

1. [**Start Shards**](#running-a-shard): Launch shard nodes on each device.
2. [**Start API**](#running-an-api): Launch the API node, one of the shards SHOULD reside in the same device.
3. [**Prepare Topology**](#prepare-topology): API discovers devices and solves for optimal layer distribution.
4. [**Load Model**](#load-model): API instructs shards to load their assigned layers.
5. [**Inference**](#chat-completions): Use `/v1/chat/completions` endpoint for generation.

See [catalog](https://github.com/firstbatchxyz/dnet/blob/master/src/dnet/api/catalog.py) for supported models.

![image](https://github.com/firstbatchxyz/dnet/blob/8a1457f907f0d38555500d3d060efbe3f5438453/dnet-tui-ss.png?raw=true)

### Viewing dnet TUI

dnet comes with a [TUI](https://github.com/firstbatchxyz/dnet-tui) built in Rust, providing a neat interface for you to load models, view the topology and chat with the loaded models.

Install the TUI with:

```sh
cargo install https://github.com/firstbatchxyz/dnet-tui.git
```

Then simply run with:

```sh
dnet-tui
```

For more details, check out the [repository](https://github.com/firstbatchxyz/dnet-tui).



### Running a Shard

Start a shard node with gRPC and HTTP ports:

```sh
uv run dnet-shard --http-port 8081 --grpc-port 58081
```

Each shard should be started on a different device and with a different port (try increment by one for each shard), like the following:

```sh
uv run dnet-shard --http-port 8082 --grpc-port 58082
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
> uv run ./scripts/prepare_model.py Qwen/Qwen3-4B-MLX-4bit
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

Response will be the otpimal topology (as given by the solver) for the discovered devices.

> [!NOTE]
>
> Once the topology is prepared, you can fetch it after via the `/topology` endpoint:
>
> ```sh
> curl http://localhost:8080/v1/topology \
>  -H "Content-Type: application/json" \
> ```

#### Load Model

Load the model on shards with prepared topology:

<!-- add devices to body here as well -->

```sh
curl -X POST http://localhost:8080/v1/load_model \
  -H "Content-Type: application/json" \
  -d $OUTPUT_FROM_PREPARE_TOPOLOGY
```

![A shard with a loaded model](https://github.com/firstbatchxyz/dnet/blob/master/dnet-shard-ss.png?raw=true)


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

## Testing

Before testing make sure to install dev path
```
uv sync --extra dev --extra mac
```

You can run Pytest tests via:

```sh
uv run pytest -v
```

You can check linting and formatting via Ruff:

```sh
# lint
uvx ruff check

# format
uvx ruff format --diff

# typecheck
uv run mypy .
```

> [!TIP]
>
> If you are using VsCode, we have prepared [tasks](./.vscode/tasks.json) that you can run easily from the <kbd> Command Palette > Tasks: Run Task </kbd>.

## Acknowledgements

**dnet** is built on top of [MLX](https://github.com/ml-explore/mlx) and inspired by pioneering work in distributed inference:

**PRIMA.CPP**: [Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters](https://arxiv.org/abs/2504.08791)

**Exo**: [Run your own AI cluster at home with everyday devices](https://github.com/exo-explore/exo)

**Petals**: [Collaborative Inference for Large Language Models](https://github.com/bigscience-workshop/petals)

## License

You can find the license [here](./LICENSE).
