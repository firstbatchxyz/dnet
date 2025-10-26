"""Utility functions for repacking model weights."""

import re
import shutil
from pathlib import Path
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
