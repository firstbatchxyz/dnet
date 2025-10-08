#!/usr/bin/env python3
import argparse
import requests
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare model for dnet serving",
        epilog="Example: uv run prepare_model.py Qwen/Qwen3-4B-MLX-4bit",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model name or HuggingFace repo ID (e.g., Qwen/Qwen3-4B-MLX-4bit)",
    )
    args = parser.parse_args()

    # call `/prepare_topology` endpoint
    print(f"Preparing topology for model: {args.model}")
    response = requests.post(
        "http://localhost:8080/v1/prepare_topology",
        json={"model": args.model},
    )
    if response.status_code == 200:
        print("Topology prepared successfully:")
        topology = response.json()
        print(json.dumps(topology, sort_keys=True, indent=2))
    else:
        print(f"Failed to prepare topology: {response.status_code}")
        print(response.text)
        exit(1)

    # call `/load_model` endpoint with the response body
    print(f"Loading: {args.model}")
    response = requests.post(
        "http://localhost:8080/v1/load_model",
        json=topology,
    )
    if response.status_code == 200:
        print("Model loaded successfully:")
        result = response.json()
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        print(f"Failed to load model: {response.status_code}")
        print(response.text)
        exit(1)
