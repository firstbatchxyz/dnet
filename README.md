<p align="center">
  <img src="https://raw.githubusercontent.com/firstbatchxyz/.github/refs/heads/master/branding/dria-logo-square.svg" alt="logo" width="168">
</p>

<p align="center">
  <h1 align="center">
    dnet
  </h1>
  <p align="center">
    <i>Distributed LLM inference on MLX using ring topology.</i>
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

## Installation

**dnet** requires several submodules, which can all be cloned with the following command:

```sh
git clone --recurse-submodules https://github.com/firstbatchxyz/dnet.git
```

**dnet** uses `uv`, so make sure it is installed. You can check for uv with the command below, and follow the [installation guide](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it.

```sh
uv --version
```

### Platform-specific MLX installation

**dnet** supports MLX on multiple platforms, but MLX is never installed by default. You must select the correct MLX variant for your system:

- **macOS (Apple Silicon):**

  ```sh
  uv sync --extra mac
  ```

- **CPU-only (Linux/Windows):**

  ```sh
  uv sync --extra cpu
  ```

- **CUDA (Linux with NVIDIA GPU):**

  ```sh
  uv sync --extra cuda
  ```

If you run just `uv sync`, MLX will NOT be installed. Always use the appropriate `--extra` flag for your platform.

After syncing dependencies, generate protos:

```sh
uv run ./scripts/generate_protos.py
```

## Usage

**dnet** uses a **dynamic topology** approach where nodes start without models, then the API discovers devices and distributes layers optimally using [distilp](https://github.com/firstbatchxyz/distilp).

1. [**Start Shards**](#running-a-shard): Launch shard nodes on each device.
2. [**Start API**](#running-an-api): Launch the API node, one of the shards SHOULD reside in the same device.
3. [**Prepare Topology**](#prepare-topology): API discovers devices and solves for optimal layer distribution.
4. [**Load Model**](#load-model): API instructs shards to load their assigned layers.
5. [**Inference**](#chat-completions): Use `/v1/chat/completions` endpoint for generation.

See [catalog](https://raw.githubusercontent.com/firstbatchxyz/dnet/refs/heads/master/src/dnet/api/catalog.py?token=GHSAT0AAAAAADJ5TKOCRPUKW5MZ6Z7Y5W4Q2JFSUQQ) for supported models.


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

## Configuration (.env)

**dnet** supports configuration via a `.env` file in the project root. This allows you to set environment variables for logging, profiling, and other runtime options without modifying code or command-line arguments.

### Example `.env`

```env
# Set logging level (e.g., DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Enable profiling (set to 1 to enable)
PROFILE=0

# Add other environment variables as needed
```

Please see `.env.example` for a complete example. The `.env` file is automatically loaded when running via `uv run` or using the provided Makefile targets. This ensures consistent configuration in both local development and CI environments.

For more details, see the relevant sections in the Makefile and CI workflow.

## Testing

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
```

> [!TIP]
>
> If you are using VsCode, we have prepared [tasks](./.vscode/tasks.json) that you can run easily from the <kbd> Command Palette > Tasks: Run Task </kbd>.

## License

You can find the license [here](./LICENSE).
