"""Utilities to repack HF shard weights into per-layer safetensors.

This enables mx.load fast-path on per-layer files for sliding_fit/offload modes.
Only assigned layers for a shard are repacked to minimize one-time work.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mlx.core as mx

from .model import ModelMetadata, get_model_metadata, load_weight, MappedFile
from .logger import logger
import json
import time


def _sanitize_model_id(model_id: str) -> str:
    s = model_id.strip().replace("\\", "/")
    # Replace non-alnum with underscores
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "model"


def _hash_layers(layers: Iterable[int]) -> str:
    h = hashlib.sha1()
    for v in sorted(int(i) for i in layers):
        h.update(str(v).encode("utf-8"))
        h.update(b",")
    return h.hexdigest()[:10]


def _copy_non_weight_artifacts(src_root: Path, dst_root: Path) -> None:
    import shutil

    dst_root.mkdir(parents=True, exist_ok=True)
    skip_exts = {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".npz",
        ".gguf",
        ".onnx",
    }
    try:
        copied = 0
        for entry in src_root.iterdir():
            if not entry.is_file():
                continue
            if entry.suffix.lower() in skip_exts:
                continue
            try:
                dst = dst_root / entry.name
                if not dst.exists():
                    shutil.copy2(entry, dst)
                    copied += 1
            except Exception:
                continue
        if copied:
            print(
                f"[repack] Copied {copied} tokenizer/config artifact(s) to {dst_root}"
            )
    except Exception:
        pass


def repack_per_layer(
    model_path: str,
    assigned_layers: List[int],
    out_root: Path,
) -> Path:
    """Repack only the assigned layers into per-layer safetensors under out_root.

    Returns the directory containing repacked files (out_root).
    """
    md: ModelMetadata = get_model_metadata(model_path)

    out_root.mkdir(parents=True, exist_ok=True)
    _copy_non_weight_artifacts(Path(md.path), out_root)

    mapped_files: Dict[str, MappedFile] = {}
    written = 0
    t0 = time.perf_counter()
    try:
        for lid in sorted(set(int(i) for i in assigned_layers)):
            layer_info = md.weight_info.get(int(lid), {})
            if not layer_info:
                continue
            tensors: Dict[str, mx.array] = {}
            for suffix, wt in layer_info.items():
                key = f"model.layers.{int(lid)}.{suffix}"
                tensors[key] = load_weight(wt, mapped_files)
            if not tensors:
                continue
            fname = out_root / f"layer_{int(lid):04d}.safetensors"
            if not fname.exists():
                mx.save_safetensors(str(fname), tensors)
                written += 1
        # API-layer tensors (embed/norm/head)
        api_subset: Dict[str, mx.array] = {}
        try:
            for k, wt in md.embed_tokens.items():
                api_subset[f"model.embed_tokens.{k}"] = load_weight(wt, mapped_files)
            for k, wt in md.norm.items():
                api_subset[f"model.norm.{k}"] = load_weight(wt, mapped_files)
            for k, wt in md.lm_head.items():
                api_subset[f"lm_head.{k}"] = load_weight(wt, mapped_files)
        except Exception:
            api_subset = {}
        if api_subset:
            api_path = out_root / "api_layers.safetensors"
            if not api_path.exists():
                mx.save_safetensors(str(api_path), api_subset)
        # Write manifest describing this repack
        try:
            manifest = {
                "version": 1,
                "model_id": model_path,
                "source_path": str(md.path),
                "assigned_layers": sorted(set(int(i) for i in assigned_layers)),
                "layers_hash": _hash_layers(assigned_layers),
                "num_layers": int(md.num_layers),
                "created_at": int(time.time()),
                "api_layers_file": "api_layers.safetensors" if api_subset else None,
                "files": [
                    f"layer_{int(i):04d}.safetensors"
                    for i in sorted(set(int(x) for x in assigned_layers))
                    if (out_root / f"layer_{int(i):04d}.safetensors").exists()
                ],
            }
            (out_root / "repack-manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )
        except Exception:
            pass
    finally:
        for mf in mapped_files.values():
            try:
                mf.mmap.close()
                mf.file.close()
            except Exception:
                pass
    try:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "[REPACK] model=%s layers=%s..%s count=%s out=%s ms=%.1f",
            model_path,
            int(min(assigned_layers)) if assigned_layers else -1,
            int(max(assigned_layers)) if assigned_layers else -1,
            int(len(set(assigned_layers))),
            str(out_root),
            dt_ms,
        )
    except Exception:
        pass
    if written:
        print(f"[repack] Wrote {written} per-layer file(s) under {out_root}")
    return out_root


def ensure_repacked_for_layers(
    model_id: str, assigned_layers: List[int]
) -> Tuple[Path, bool]:
    """Create a deterministic per-model, per-assignment output directory and repack if needed.

    Returns the directory path containing repacked files.
    """
    safe = _sanitize_model_id(model_id)
    layer_hash = _hash_layers(assigned_layers)
    base_dir = Path(os.getenv("DNET_REPACK_DIR", "repacked_models"))
    out_root = base_dir / safe / layer_hash
    # Quick existence check: if at least one expected file exists, assume done
    expected = (
        out_root / f"layer_{int(sorted(set(assigned_layers))[0]):04d}.safetensors"
    )
    repacked = False
    if not expected.exists():
        repack_per_layer(model_id, assigned_layers, out_root)
        repacked = True
    # Ensure manifest exists even when files already present
    try:
        man_path = out_root / "repack-manifest.json"
        if not man_path.exists():
            md = get_model_metadata(model_id)
            manifest = {
                "version": 1,
                "model_id": model_id,
                "source_path": str(md.path),
                "assigned_layers": sorted(set(int(i) for i in assigned_layers)),
                "layers_hash": layer_hash,
                "num_layers": int(md.num_layers),
                "created_at": int(time.time()),
                "api_layers_file": (
                    "api_layers.safetensors"
                    if (out_root / "api_layers.safetensors").exists()
                    else None
                ),
                "files": [p.name for p in sorted(out_root.glob("layer_*.safetensors"))],
            }
            man_path.write_text(json.dumps(manifest, indent=2))
    except Exception:
        pass
    return out_root, repacked


def delete_repacked_layers(
    *,
    model_id: str | None = None,
    all_flag: bool = False,
    base_dir: str | Path | None = None,
    current_model_path: str | None = None,
) -> list[str]:
    """Delete repacked per-layer buckets under the base directory.

    Args:
        model_id: Optional HF repo id (sanitized) to target a specific bucket
        all_flag: When True, remove the entire base repack directory
        base_dir: Base directory; defaults to env DNET_REPACK_DIR or 'repacked_models'
        current_model_path: Optional current model path/id used when model_id is None

    Returns:
        List of removed paths (as strings)
    """
    import shutil

    if base_dir is None:
        base_dir = os.getenv("DNET_REPACK_DIR", "repacked_models")
    base = Path(base_dir)
    removed: list[str] = []

    if all_flag:
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
            removed.append(str(base))
        return removed

    # Remove a specific model bucket by id if provided
    if model_id:
        safe = _sanitize_model_id(str(model_id))
        target = base / safe
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            removed.append(str(target))
        return removed

    # Default: if current_model_path is provided, attempt to remove its bucket.
    # Handle three cases robustly:
    #  1) If a repack manifest exists under the path, read the original model_id
    #     and delete base_dir/<sanitize(model_id)>.
    #  2) If the path is inside base_dir, delete the bucket root base_dir/<safe>.
    #  3) Fallback to previous behavior (sanitize the path string).
    if current_model_path:
        try:
            pcur = Path(current_model_path)

            # Case 1: manifest with original model_id
            try:
                man_path = pcur / "repack-manifest.json"
                if man_path.exists():
                    manifest = json.loads(man_path.read_text())
                    src_id = manifest.get("model_id")
                    if src_id:
                        safe = _sanitize_model_id(str(src_id))
                        target = base / safe
                        if target.exists():
                            shutil.rmtree(target, ignore_errors=True)
                            removed.append(str(target))
                            return removed
            except Exception:
                pass

            # Case 2: current path is within base repack directory
            try:
                pb = base.resolve()
                pr = pcur.resolve()
                rel = pr.relative_to(pb)
                parts = rel.parts
                if parts:
                    bucket = pb / parts[0]
                    if bucket.exists():
                        shutil.rmtree(bucket, ignore_errors=True)
                        removed.append(str(bucket))
                        return removed
            except Exception:
                pass

            # Case 3: Fallback to sanitizing the provided path string
            try:
                safe = _sanitize_model_id(str(current_model_path))
                target = base / safe
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                    removed.append(str(target))
            except Exception:
                pass
        except Exception:
            pass
    return removed
