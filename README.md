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

## Usage

To use dnet, start multiple [shards](#running-a-shard) and a single [API](#running-an-api). It supports the following models

- Qwen3
- DeepSeek V2
- MLX formats: fp16, bf16, 4-bit, 8-bit quantized

### Running an API

### Running a Shard

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
