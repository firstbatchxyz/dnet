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
import re
from pathlib import Path
import shutil
from typing import Dict, List

import mlx.core as mx
from dnet.utils.model import get_model_metadata, load_weight, MappedFile

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def chunk_layers(layer_list: List[int], window_size: int) -> List[List[int]]:
    return [
        layer_list[i : i + window_size] for i in range(0, len(layer_list), window_size)
    ]


def repack_windows(
    model_dir: Path, chunks: List[List[int]], out_prefix: Path
) -> List[Path]:
    # Resolve model and use metadata + memory mapping to load only needed tensors
    md = get_model_metadata(str(model_dir))
    mapped_files: Dict[str, MappedFile] = {}
    written: List[Path] = []
    # Ensure the destination root contains the correct config.json from the source model
    try:
        out_root = out_prefix.parent
        out_root.mkdir(parents=True, exist_ok=True)
        src_cfg = Path(md.path) / "config.json"
        dst_cfg = out_root / "config.json"
        if src_cfg.exists():
            # Always copy the source config to avoid stale/mismatched configs
            shutil.copy2(src_cfg, dst_cfg)
            print(f"Copied config.json to {dst_cfg}")
    except Exception as e:
        print(f"Warning: failed to place config.json: {e}")
    try:
        for chunk in chunks:
            if not chunk:
                continue
            subset: Dict[str, mx.array] = {}
            for layer_idx in chunk:
                layer_info = md.weight_info.get(int(layer_idx), {})
                if not layer_info:
                    continue
                for suffix, wt in layer_info.items():
                    key = f"model.layers.{int(layer_idx)}.{suffix}"
                    subset[key] = load_weight(wt, mapped_files)
            if not subset:
                continue
            a, b = int(chunk[0]), int(chunk[-1])
            out_name = f"{out_prefix.name}_{a:03d}-{b:03d}.safetensors"
            out_dir = out_prefix.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / out_name
            mx.save_safetensors(str(out_path), subset)
            written.append(out_path)
            print(f"Wrote {out_path} with {len(subset)} tensors")
    finally:
        for mf in mapped_files.values():
            try:
                mf.mmap.close()
                mf.file.close()
            except Exception:
                pass
    return written


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
