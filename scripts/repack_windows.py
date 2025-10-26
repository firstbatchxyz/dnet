#!/usr/bin/env python3
"""
Repack model weights into per-window safetensors files (single-list input).

Usage:
  uv run python scripts/repack_windows.py \
    --model-dir Qwen/Qwen3-32B-MLX-bf16 \
    --layers "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]" \
    --window-size 10 \
    --out-prefix ./shard_weights/qwen3_shard0

Notes:
  - Accepts exactly one Python list for --layers and chunks it by --window-size
    preserving order. This matches shard layer assignment order.
  - Resolves --model-dir like runtime (supports HF repo ids).
  - Loads only tensors for selected layers using memory-mapped I/O.
  - Saves keys as "model.layers.<idx>.<name>" for compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dnet.utils.repack import chunk_layers, repack_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Model directory or HF repo id (resolved like runtime)",
    )
    ap.add_argument(
        "--layers",
        required=True,
        help="Single Python list of layer indices (e.g., '[0,1,2,...]')",
    )
    ap.add_argument(
        "--window-size",
        type=int,
        required=True,
        help="Chunk size for the provided list of layers",
    )
    ap.add_argument("--out-prefix", required=True, help="Prefix for output files")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_prefix = Path(args.out_prefix)
    import ast

    lst = ast.literal_eval(args.layers)
    if not isinstance(lst, list) or not all(isinstance(x, int) for x in lst):
        raise SystemExit("--layers must be a Python list of integers")
    if args.window_size <= 0:
        raise SystemExit("--window-size must be > 0")

    chunks = chunk_layers(lst, args.window_size)
    repack_windows(model_dir, chunks, out_prefix)


if __name__ == "__main__":
    main()
