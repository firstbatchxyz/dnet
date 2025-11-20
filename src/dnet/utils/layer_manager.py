"Layer manager for memory-mapped LLM layers with prefetching on macOS."

import ctypes
import ctypes.util
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import mlx.core as mx

from .model import (
    MappedFile,
    ModelMetadata,
    get_model_layer_name,
    load_weight,
)
from .logger import logger

# Load libc for madvise
libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.madvise.restype = ctypes.c_int

# macOS madvise constants
MADV_SEQUENTIAL = 2  # Sequential access
MADV_WILLNEED = 3  # Prefetch pages
MADV_DONTNEED = 4  # Pages not needed


# Prefetch policy control (config-driven)
# Modes:
#   - "full": SEQUENTIAL + WILLNEED
#   - "sequential": SEQUENTIAL only (no WILLNEED)
#   - "off": no madvise at all
_VALID_PREFETCH_MODES = {"full", "sequential", "off"}


class LayerManager:
    """Manages memory-mapped LLM layers with prefetching. This is built around
    the assumption that API servers handle the embedding and output projection.
    Since embedding tokens and LM heads are generally 1GB in size even for
    big models such as Llama-65B, it can fit in RAM, and hence does not
    require memory mapping. This class is meant to be used by the shard servers
    which hold the layers."""

    def __init__(
        self,
        model_metadata: ModelMetadata,
        assigned_layers: List[int],
        thread_pool_size: int = 2,
        *,
        use_mxload_fastpath: bool = False,
        prefetch_mode: str = "off",
    ):
        """
        Args:
            model_metadata: Model metadata
            layer_info: Layer indexes
            prefetch_queue_size: Number of layers to prefetch ahead
            thread_pool_size: Number of threads dedicated to prefetching
        """
        # Extract the name of safetensors associated with the layers
        self.assigned_layers = set(assigned_layers)

        # Weight metadata
        self.weight_info = model_metadata.weight_info
        # Skip pre-mapping in mx.load fast-path; map files only in mmap mode
        if use_mxload_fastpath:
            self.mapped_files = {}
        else:
            filenames = set(
                wts.filename
                for layer in assigned_layers
                for wts in self.weight_info[layer].values()
            )
            self.mapped_files = {fname: MappedFile(fname) for fname in filenames}

        self.executor = ThreadPoolExecutor(max_workers=thread_pool_size)
        # Only enable mx.load fast-path for explicitly repacked windows.
        # Default is False to avoid loading entire multi-layer shard files.
        self._use_mxload_fastpath = bool(use_mxload_fastpath)
        # Config-driven prefetch mode
        pm = (prefetch_mode or "off").strip().lower()
        self._prefetch_mode = pm if pm in _VALID_PREFETCH_MODES else "off"
        logger.info(f"Initialized LLM manager with layers {self.assigned_layers}")

    def _memadvise_layer(self, layer_idx: int, memadvise: int) -> bool:
        if self._use_mxload_fastpath:
            return True
        if layer_idx not in self.assigned_layers:
            return False

        # get information about tensors that this layer needs
        weight_data = self.weight_info[layer_idx]

        # loop over each tensor, and prefetch
        success = True
        for wt in weight_data.values():
            offset, size = wt.offset, wt.size_bytes
            mapped_file = self.mapped_files[wt.filename]
            layer_addr = mapped_file.base_addr + offset

            # Use madvise to prefetch
            result = libc.madvise(layer_addr, size, memadvise)
            success &= True if result == 0 else False
        return success

    def prefetch_layer(self, layer_idx: int):
        """Prefetch a layer's weights using madvise with a coalesced range per file.

        For repacked models, all tensors for a given layer live in a single
        safetensors file and occupy a mostly contiguous span. Coalescing the
        advisory into one (SEQUENTIAL+WILLNEED) per file dramatically reduces
        syscall overhead and improves kernel readahead behavior on macOS.
        Falls back to per-tensor hints when needed.
        """
        import time as _time

        t0 = _time.perf_counter()
        if self._use_mxload_fastpath:
            # Read-ahead the per-layer file to warm OS cache; overlaps IO with compute
            try:
                info = self.weight_info.get(layer_idx, {})
                fnames = {wt.filename for wt in info.values()}
                if not fnames:
                    return False
                fname = next(iter(fnames))
                chunk = 4 * 1024 * 1024
                with open(fname, "rb", buffering=0) as f:
                    while True:
                        b = f.read(chunk)
                        if not b:
                            break
                dt_ms = (_time.perf_counter() - t0) * 1000.0
                logger.info(
                    f"[PROFILE][PREFETCH-READ] layer={layer_idx} mode=mxload ms={dt_ms:.2f}"
                )
                return True
            except Exception:
                dt_ms = (_time.perf_counter() - t0) * 1000.0
                logger.info(
                    f"[PROFILE][PREFETCH-READ] layer={layer_idx} mode=mxload ms={dt_ms:.2f} (failed)"
                )
                return False
        mode = self._prefetch_mode
        if mode == "off":
            # Skip any OS advice; treat as success for control flow
            dt_ms = (_time.perf_counter() - t0) * 1000.0
            logger.info(
                f"[PROFILE][PREFETCH] layer={layer_idx} mode=off ms={dt_ms:.2f}"
            )
            return True

        weight_data = self.weight_info[layer_idx]

        # Group tensor extents by filename and coalesce [min_offset, max_end)
        by_file: Dict[str, tuple[int, int]] = {}
        try:
            for wt in weight_data.values():
                start = wt.offset
                end = wt.offset + wt.size_bytes
                cur = by_file.get(wt.filename)
                if cur is None:
                    by_file[wt.filename] = (start, end)
                else:
                    by_file[wt.filename] = (min(cur[0], start), max(cur[1], end))

            ok = True
            for fname, (start, end) in by_file.items():
                mapped_file = self.mapped_files.get(fname)
                if mapped_file is None:
                    # Fallback: per-tensor advise if file isn't mapped yet
                    if mode in ("full", "sequential"):
                        ok &= self._memadvise_layer(layer_idx, MADV_SEQUENTIAL)
                    if mode == "full":
                        ok &= self._memadvise_layer(layer_idx, MADV_WILLNEED)
                    continue

                base_addr = mapped_file.base_addr + start
                size = end - start
                # Hint sequential, and optionally will-need on the coalesced span
                r_seq = libc.madvise(base_addr, size, MADV_SEQUENTIAL)
                if mode == "full":
                    r_w = libc.madvise(base_addr, size, MADV_WILLNEED)
                    ok &= r_seq == 0 and r_w == 0
                else:
                    ok &= r_seq == 0

            dt_ms = (_time.perf_counter() - t0) * 1000.0
            if ok:
                logger.info(
                    f"[PROFILE][PREFETCH] layer={layer_idx} mode={mode} ms={dt_ms:.2f}"
                )
            else:
                logger.info(
                    f"[PROFILE][PREFETCH] layer={layer_idx} mode={mode} ms={dt_ms:.2f} (partial)"
                )
            return ok
        except Exception:
            # Robust fallback: original per-tensor path
            result = True
            if mode in ("full", "sequential"):
                _ = self._memadvise_layer(layer_idx, MADV_SEQUENTIAL)
            if mode == "full":
                result = self._memadvise_layer(layer_idx, MADV_WILLNEED)
            dt_ms = (_time.perf_counter() - t0) * 1000.0
            if result:
                logger.info(
                    f"[PROFILE][PREFETCH] layer={layer_idx} mode={mode} ms={dt_ms:.2f}"
                )
                return True
            else:
                logger.info(
                    f"[PROFILE][PREFETCH] layer={layer_idx} mode={mode} ms={dt_ms:.2f} (failed)"
                )
                return False

    def release_layer(self, layer_idx: int):
        """Mark layer as not needed anymore"""
        if self._use_mxload_fastpath:
            return True
        result = self._memadvise_layer(layer_idx, MADV_DONTNEED)
        if result:
            logger.info(f"Released layer {layer_idx}")
            return True
        else:
            logger.info(f"Failed to release layer {layer_idx}")
            return False

    def load_layer_to_gpu(self, layer_idx) -> Dict[str, mx.array]:
        """Load layer from memory map to GPU"""

        if layer_idx not in self.assigned_layers:
            raise RuntimeError(f"layer {layer_idx} not assigned to this node")

        # get information about tensors that this layer needs
        weight_data = self.weight_info[layer_idx]
        data: Dict[str, mx.array] = {}

        # Fast-path using per-file mx.load: if a layer's tensors are all in one
        # file (repacked window) OR eager-load cache is present, fetch arrays
        # by key from the file dict(s). This avoids per-tensor BF16 conversions.
        try:
            info = self.weight_info[layer_idx]
            fnames = {wt.filename for wt in info.values()}
        except Exception:
            fnames = set()
            info = {}

        if self._use_mxload_fastpath and info and (len(fnames) == 1):
            try:
                collected = 0
                for fname in fnames:
                    d = mx.load(fname)

                    if not isinstance(d, dict):
                        raise RuntimeError("mx.load did not return a dict")

                    p1 = f"model.layers.{layer_idx}."
                    p2 = f"layers.{layer_idx}."
                    for k, v in d.items():
                        if k.startswith(p1):
                            suffix = k[len(p1) :]
                        elif k.startswith(p2):
                            suffix = k[len(p2) :]
                        else:
                            continue
                        data[get_model_layer_name(layer_idx, suffix)] = v
                        collected += 1
                if collected == 0:
                    data.clear()
                    raise RuntimeError("mxload keys not found for layer")
                mx.eval(*data.values())
                return data
            except Exception:
                data.clear()

        # Default path: mmap-based per-tensor load
        for name, wt in weight_data.items():
            data[get_model_layer_name(layer_idx, name)] = load_weight(
                wt, self.mapped_files
            )
        return data

    def async_prefetch(self, layer_idx):
        """Asynchronously prefetch a layer"""
        return self.executor.submit(self.prefetch_layer, layer_idx)

    def close(self):
        self.executor.shutdown(wait=True)
        for mapped_file in self.mapped_files.values():
            mapped_file.mmap.close()
            mapped_file.file.close()
