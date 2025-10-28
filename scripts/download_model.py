#!/usr/bin/env python3
"""Download a model from HuggingFace to local cache."""

import argparse
import sys
from pathlib import Path

from mlx_lm.utils import get_model_path


def download_model(repo_id: str) -> None:
    """Download a model from HuggingFace.

    Args:
        repo_id: Model name or HuggingFace repo ID (e.g., Qwen/Qwen3-4B-MLX-4bit)
    """
    print(f"Downloading model: {repo_id}")
    print("This may take a while depending on the model size...")

    try:
        # get_model_path will download if not available locally
        path_result = get_model_path(repo_id)

        # Handle tuple return (mlx-lm version compatibility)
        if isinstance(path_result, tuple):
            path = path_result[0] if path_result else None
        else:
            path = path_result

        if path is None:
            print(f"Error: Could not resolve model path for {repo_id}")
            sys.exit(1)

        path = Path(path)
        print(f"\nModel downloaded successfully at: {path}")

        safetensors = list(path.glob("*.safetensors"))
        if safetensors:
            total_size = sum(f.stat().st_size for f in safetensors)
            size_gb = total_size / (1024**3)
            print(f"Size: {size_gb:.2f} GB ({len(safetensors)} safetensors files)")

    except Exception as e:
        print(f"\n Could not download model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a model from HuggingFace to local cache",
        epilog="Example: uv run download_model.py Qwen/Qwen3-4B-MLX-4bit",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model name or HuggingFace repo ID (e.g., Qwen/Qwen3-4B-MLX-4bit)",
    )
    args = parser.parse_args()

    download_model(args.model)
