#!/usr/bin/env python3
"""
Prepare and load model for Context Parallelism (CP).

Unlike ring/pipeline parallelism where each shard gets non-overlapping layers,
CP loads ALL layers on ALL shards. Each shard processes a portion of the
context window (sequence dimension) while maintaining the full model.

Usage:
    uv run scripts/prepare_cp_model.py Qwen/Qwen3-4B-MLX-4bit
    uv run scripts/prepare_cp_model.py Qwen/Qwen3-4B-MLX-4bit --shards m4s1,m4s2

The ModelManager will automatically assign CP ranks based on device order:
    - rank 0: first device in list
    - rank 1: second device in list
    - etc.

For two-device CP, each device handles half the context window.
"""

import argparse
import json
import sys
from typing import Literal

import requests

from dnet.api.models import ManualDevice, PrepareTopologyManualRequest
from dnet.core.types.topology import LayerAssignment

from scripts.cp_utils import get_kv_bits_from_server, get_devices


def get_model_config(model: str) -> dict:
    """Fetch model config from HuggingFace to get num_layers."""
    try:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=model,
            filename="config.json",
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not fetch model config from HuggingFace: {e}")
        return {}


def prepare_cp_topology(
    api_url: str,
    model: str,
    devices: list[ManualDevice],
    num_layers: int,
    seq_len: int,
    kv_bits: Literal["4bit", "8bit", "fp16"],
) -> dict:
    """Prepare manual topology for CP mode (all shards get all layers)."""
    all_layers = list(range(num_layers))

    # For CP, each device gets ALL layers (full model replication)
    assignments: list[LayerAssignment] = []
    for i, device in enumerate(devices):
        next_idx = (i + 1) % len(devices)
        next_instance = devices[next_idx].instance

        assignments.append(
            LayerAssignment(
                instance=device.instance,
                layers=[all_layers],
                window_size=num_layers,
                residency_size=num_layers,
                next_instance=next_instance,
            )
        )

    request = PrepareTopologyManualRequest(
        model=model,
        devices=devices,
        assignments=assignments,
        num_layers=num_layers,
        kv_bits=kv_bits,
    )

    response = requests.post(
        f"{api_url}/v1/prepare_topology_manual",
        json=request.model_dump(),
    )
    response.raise_for_status()
    return response.json()


def load_model(api_url: str, model: str) -> dict:
    """Load model on all shards."""
    response = requests.post(f"{api_url}/v1/load_model", json={"model": model})
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and load model for Context Parallelism",
        epilog="""

Examples:
    # Auto-discover all shards and use them for CP
    uv run scripts/prepare_cp_model.py Qwen/Qwen3-4B-MLX-4bit

    # Use specific shards for CP
    uv run scripts/prepare_cp_model.py Qwen/Qwen3-4B-MLX-4bit --shards m4s1,m4s2

    # Use custom API URL
    uv run scripts/prepare_cp_model.py Qwen/Qwen3-4B-MLX-4bit --api http://10.0.0.1:8080
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model name or HuggingFace repo ID (e.g., Qwen/Qwen3-4B-MLX-4bit)",
    )
    parser.add_argument(
        "--api",
        type=str,
        default="http://localhost:8080",
        help="API server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--shards",
        type=str,
        default=None,
        help="Comma-separated shard instance names (default: all available)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length (default: from model config or 8192)",
    )
    args = parser.parse_args()

    api_url = args.api.rstrip("/")

    # Get kv_bits from server settings
    kv_bits = get_kv_bits_from_server(api_url)

    # Step 1: Discover devices
    print(f"[1/4] Fetching available devices from {api_url}...")
    try:
        devices_dict = get_devices(api_url)
    except requests.RequestException as e:
        print(f"Error: Could not connect to API at {api_url}: {e}")
        sys.exit(1)

    # Build typed ManualDevice list, filtering out managers
    all_devices: list[ManualDevice] = []
    for instance, props in devices_dict.items():
        if props.is_manager:
            continue
        all_devices.append(
            ManualDevice(
                instance=instance,
                local_ip=props.local_ip,
                server_port=props.server_port,
                shard_port=props.shard_port,
            )
        )

    if not all_devices:
        print("Error: No shards available. Make sure shard nodes are running.")
        sys.exit(1)

    # Filter by requested shards if specified
    shards = all_devices
    if args.shards:
        requested = set(args.shards.split(","))
        shards = [d for d in all_devices if d.instance in requested]
        if not shards:
            print(f"Error: None of the requested shards found: {args.shards}")
            print(f"Available: {[d.instance for d in all_devices]}")
            sys.exit(1)

    print(f"    Using {len(shards)} shard(s) for Context Parallelism:")
    for i, s in enumerate(shards):
        print(f"      [{i}] {s.instance} ({s.local_ip}:{s.server_port})")

    # Step 2: Get model config
    print(f"[2/4] Fetching model config for {args.model}...")
    model_config = get_model_config(args.model)

    num_layers = model_config.get("num_hidden_layers") or model_config.get("n_layers")
    if not num_layers:
        print("Error: Could not determine number of layers from model config.")
        sys.exit(1)

    print(f"    Model has {num_layers} layers (full model on each shard)")

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = model_config.get("max_position_embeddings") or 8192
    print(f"    Sequence length: {seq_len}")

    # Step 3: Prepare topology
    print("[3/4] Preparing CP topology...")
    try:
        topology = prepare_cp_topology(
            api_url=api_url,
            model=args.model,
            devices=shards,
            num_layers=num_layers,
            seq_len=seq_len,
            kv_bits=kv_bits,
        )
        print("    Topology prepared successfully")
        print(f"    Model: {topology.get('model')}")
        assignments = topology.get("assignments", [])
        print(f"    Devices: {[a.get('instance') for a in assignments]}")
    except requests.RequestException as e:
        print(f"Error: Failed to prepare topology: {e}")
        sys.exit(1)

    # Step 4: Load model
    print("[4/4] Loading model on all shards (this may take a while)...")
    try:
        result = load_model(api_url, args.model)
        print("    Model loaded successfully!")
        print()
        print("=" * 60)
        print("Context Parallelism Ready")
        print("=" * 60)
        print(f"  Model:      {args.model}")
        print(f"  CP Ranks:   {len(shards)}")
        print(f"  Shards:     {', '.join(s.instance for s in shards)}")
        print(f"  KV Bits:    {kv_bits}")
        print(f"  Seq Len:    {seq_len}")
        print()
        print(f"Each shard has the full model and will process 1/{len(shards)} of")
        print("the context window during inference.")
        print()

        for status in result.get("shard_statuses", []):
            success = "✓" if status.get("success") else "✗"
            print(
                f"  {success} {status.get('instance')}: {status.get('message', 'OK')}"
            )

    except requests.RequestException as e:
        print(f"Error: Failed to load model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
