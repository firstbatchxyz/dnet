"Layer manager for memory-mapped LLM layers with prefetching on macOS."

import ctypes
import ctypes.util
import fcntl as _fcntl
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, List
from collections import OrderedDict

import mlx.core as mx
import numpy as np

from .model import (
    MappedFile,
    ModelMetadata,
    get_model_layer_name,
    load_weight,
)
from .serialization import safetensor_dtype_map
from .logger import logger

# Load libc for madvise
libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.madvise.restype = ctypes.c_int

# macOS madvise constants
MADV_SEQUENTIAL = 2  # Sequential access
MADV_WILLNEED = 3  # Prefetch pages
MADV_DONTNEED = 4  # Pages not needed


# Prefetch policy control
# Modes:
#   - "full": SEQUENTIAL + WILLNEED (default)
#   - "sequential": SEQUENTIAL only (no WILLNEED)
#   - "off": no madvise at all
_VALID_PREFETCH_MODES = {"full", "sequential", "off"}
_PREFETCH_MODE = os.getenv("RING_PREFETCH_MODE", "full").strip().lower()
if _PREFETCH_MODE not in _VALID_PREFETCH_MODES:
    _PREFETCH_MODE = "full"


def get_prefetch_mode() -> str:
    return _PREFETCH_MODE


def set_prefetch_mode(mode: str) -> str:
    global _PREFETCH_MODE
    m = (mode or "").strip().lower()
    if m in _VALID_PREFETCH_MODES:
        _PREFETCH_MODE = m
    return _PREFETCH_MODE


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
        file_cache_mode: str | None = None,
        file_cache_cap: int | None = None,
        eager_load: bool | None = None,
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

        # Open memory-mapped file
        self.weight_info = weight_info = model_metadata.weight_info
        filenames = set(
            wts.filename
            for layer in assigned_layers
            for wts in weight_info[layer].values()
        )
        self.mapped_files = {fname: MappedFile(fname) for fname in filenames}

        self.executor = ThreadPoolExecutor(max_workers=thread_pool_size)
        self._file_cache: "OrderedDict[str, Dict[str, mx.array]]" = OrderedDict()
        self._file_cache_lock = threading.Lock()
        # Cache mode: 'none' disables caching; otherwise LRU with capacity tied to RING_RESIDENT_WINDOWS
        mode = (file_cache_mode or "auto").strip().lower()
        self._file_cache_mode = mode
        # Capacity: explicit cap or default to 1
        cap = max(1, int(file_cache_cap or 1))
        self._file_cache_cap = cap
        try:
            logger.info(
                f"[FILE-CACHE] mode={self._file_cache_mode} cap={self._file_cache_cap}"
            )
        except Exception:
            pass

        logger.info(f"Initialized LLM manager with layers {self.assigned_layers}")

        # Optional eager-load: for fits-in-memory cases, pre-load all files once.
        eager = bool(eager_load)
        if eager and self._file_cache_mode != "none":
            for fname in filenames:
                try:
                    key = os.path.abspath(fname)
                    if len(self._file_cache) >= self._file_cache_cap:
                        # Respect capacity
                        logger.info(
                            f"[EAGER-LOAD] capacity reached (cap={self._file_cache_cap}); skipping {key}"
                        )
                        break
                    d = mx.load(fname)
                    self._file_cache[key] = d
                    logger.info(
                        f"[EAGER-LOAD] loaded {key} tensors={len(d)} cap={self._file_cache_cap}"
                    )
                except Exception as e:
                    logger.warning(f"[EAGER-LOAD] failed {fname}: {e}")

    def _memadvise_layer(self, layer_idx: int, memadvise: int) -> bool:
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
        mode = get_prefetch_mode()
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
        # Skipping heavy DONTNEED advice can significantly reduce tail latency
        # on direct-I/O paths where the page cache is not used.

        try:
            fio_direct = os.getenv("RING_FILE_IO", "").strip().lower() == "direct"
        except Exception:
            fio_direct = False

        try:
            release_mode = (
                (os.getenv("RING_RELEASE_MODE", "advice") or "advice").strip().lower()
            )
        except Exception:
            release_mode = "advice"

        if fio_direct or release_mode in {"none", "off", "skip"}:
            # Nothing to advise; model/unload + cache eviction free real memory
            logger.info(
                f"Release skip for layer {layer_idx} (mode={release_mode or 'advice'}, direct_io={fio_direct})"
            )
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

        if info and (
            len(fnames) == 1
            or bool(self._file_cache)
            or self._file_cache_mode == "none"
        ):
            try:
                # Determine path: cache or no-cache mode
                use_cache = self._file_cache_mode != "none"
                # Ensure file dicts for all contributing files are available
                for fname in fnames:
                    key = os.path.abspath(fname)
                    if use_cache:
                        with self._file_cache_lock:
                            present = key in self._file_cache
                        if not present:
                            d_new = mx.load(fname)
                            with self._file_cache_lock:
                                # Evict LRU if over capacity
                                while len(self._file_cache) >= self._file_cache_cap:
                                    try:
                                        evk, _ = self._file_cache.popitem(last=False)
                                        try:
                                            state_files = [
                                                os.path.basename(k)
                                                for k in self._file_cache.keys()
                                            ]
                                            logger.info(
                                                f"[FILE-CACHE] action=evict file={os.path.basename(evk)} cap={self._file_cache_cap} size={len(self._file_cache)} files={state_files}"
                                            )
                                        except Exception:
                                            logger.info(
                                                f"[FILE-CACHE] action=evict file={evk} cap={self._file_cache_cap}"
                                            )
                                    except Exception:
                                        break
                                self._file_cache[key] = d_new
                                try:
                                    state_files = [
                                        os.path.basename(k)
                                        for k in self._file_cache.keys()
                                    ]
                                    logger.info(
                                        f"[FILE-CACHE] action=load file={os.path.basename(key)} tensors={len(d_new)} size={len(self._file_cache)}/{self._file_cache_cap} files={state_files}"
                                    )
                                except Exception:
                                    logger.info(
                                        f"[FILE-CACHE] action=load file={key} tensors={len(d_new)} size={len(self._file_cache)}/{self._file_cache_cap}"
                                    )
                        else:
                            # Touch LRU order
                            with self._file_cache_lock:
                                self._file_cache.move_to_end(key)
                    else:
                        # No-cache: load ephemeral file dict
                        pass
                # Robust collection: scan file dict(s) for keys matching this layer
                collected = 0
                for fname in fnames:
                    key = os.path.abspath(fname)
                    if use_cache:
                        with self._file_cache_lock:
                            d = self._file_cache.get(key)
                        if d is None:
                            continue
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
                    else:
                        # No-cache: load file dict transiently
                        d = mx.load(fname)
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
                # If nothing matched (unexpected), fall back to per-tensor path
                if collected == 0:
                    data.clear()
                    raise RuntimeError("mxload keys not found for layer")
                try:
                    if os.getenv("RING_DEBUG_MATERIALIZE_PATH", "").strip():
                        path_label = "mxload-cache" if use_cache else "mxload-nocache"
                        logger.info(
                            "[MATERIALIZE-PATH] layer=%s path=%s files=%s tensors=%s",
                            layer_idx,
                            path_label,
                            [os.path.basename(f) for f in sorted(fnames)],
                            collected,
                        )
                except Exception:
                    pass
                return data
            except Exception:
                # Fall back to default path on any error
                data.clear()

        # Direct I/O (coalesced per-file) optionally enabled via env
        use_direct = False
        try:
            use_direct = os.getenv("RING_FILE_IO", "").strip().lower() == "direct"
        except Exception:
            use_direct = False

        if not use_direct:
            # Default path: mmap-based per-tensor load
            for name, wt in weight_data.items():
                data[get_model_layer_name(layer_idx, name)] = load_weight(
                    wt, self.mapped_files
                )
            try:
                if os.getenv("RING_DEBUG_MATERIALIZE_PATH", "").strip():
                    logger.info(
                        f"[MATERIALIZE-PATH] layer={layer_idx} path=mmap files={sorted({w.filename for w in weight_data.values()})}"
                    )
            except Exception:
                pass
            return data

        # Direct path modes: coalesce (default) or per_tensor (reduced peak memory).
        mode = (os.getenv("RING_DIRECT_MODE", "coalesce") or "coalesce").strip().lower()

        files: Dict[str, List[tuple[str, any]]] = {}
        for name, wt in weight_data.items():
            files.setdefault(wt.filename, []).append((name, wt))

        for fname, items in files.items():
            try:
                fd = os.open(fname, os.O_RDONLY)
            except Exception:
                # Fallback to buffered path for this file
                for name, wt in items:
                    data[get_model_layer_name(layer_idx, name)] = load_weight(
                        wt, self.mapped_files
                    )
                continue

            try:
                try:
                    _fcntl.fcntl(fd, _fcntl.F_NOCACHE, 1)
                except Exception:
                    pass

                if mode == "per_tensor":
                    # Read each tensor separately to avoid a large coalesced blob
                    for name, wt in items:
                        try:
                            t_bytes = os.pread(fd, int(wt.size_bytes), int(wt.offset))
                            if not t_bytes or len(t_bytes) < int(wt.size_bytes):
                                raise OSError("short read")
                            mv = memoryview(t_bytes)
                            if wt.dtype == "BF16":
                                uint16_data = np.frombuffer(mv, dtype=np.uint16)
                                float32_data = (
                                    uint16_data.astype(np.uint32) << 16
                                ).view(np.float32)
                                arr = (
                                    mx.array(float32_data)
                                    .reshape(wt.shape)
                                    .astype(mx.bfloat16)
                                )
                            else:
                                np_dt = safetensor_dtype_map[wt.dtype]
                                np_data = np.frombuffer(mv, dtype=np_dt)
                                arr = mx.array(np_data).reshape(wt.shape)
                            data[get_model_layer_name(layer_idx, name)] = arr
                        except Exception:
                            # Fallback per-tensor via mmap path on failure
                            data[get_model_layer_name(layer_idx, name)] = load_weight(
                                wt, self.mapped_files
                            )
                    continue

                # Default: coalesced span per file (efficient syscalls)
                min_off = int(min(wt.offset for _, wt in items))
                max_end = int(max(wt.offset + wt.size_bytes for _, wt in items))
                span_size = int(max_end - min_off)

                align = 4096
                aligned_start = (min_off // align) * align
                pad = int(min_off - aligned_start)
                read_size = int(pad + span_size)

                try:
                    blob = os.pread(fd, read_size, aligned_start)
                except OSError:
                    blob = None

                if not blob or len(blob) < read_size:
                    # Fallback: per-tensor via mmap path
                    for name, wt in items:
                        data[get_model_layer_name(layer_idx, name)] = load_weight(
                            wt, self.mapped_files
                        )
                    continue

                mv = memoryview(blob)
                for name, wt in items:
                    off0 = int((wt.offset - aligned_start))
                    off1 = off0 + int(wt.size_bytes)
                    t_bytes = mv[off0:off1]
                    if wt.dtype == "BF16":
                        uint16_data = np.frombuffer(t_bytes, dtype=np.uint16)
                        float32_data = (uint16_data.astype(np.uint32) << 16).view(
                            np.float32
                        )
                        arr = (
                            mx.array(float32_data).reshape(wt.shape).astype(mx.bfloat16)
                        )
                    else:
                        np_dt = safetensor_dtype_map[wt.dtype]
                        np_data = np.frombuffer(t_bytes, dtype=np_dt)
                        arr = mx.array(np_data).reshape(wt.shape)
                    data[get_model_layer_name(layer_idx, name)] = arr
            finally:
                try:
                    os.close(fd)
                except Exception:
                    pass

        try:
            if os.getenv("RING_DEBUG_MATERIALIZE_PATH", "").strip():
                logger.info(
                    f"[MATERIALIZE-PATH] layer={layer_idx} path=direct files={sorted({w.filename for w in weight_data.values()})}"
                )
        except Exception:
            pass
        return data

    def async_prefetch(self, layer_idx):
        """Asynchronously prefetch a layer"""
        return self.executor.submit(self.prefetch_layer, layer_idx)

    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)

        for mapped_file in self.mapped_files.values():
            mapped_file.mmap.close()
            mapped_file.file.close()
        self.executor.shutdown(wait=True)

        for mapped_file in self.mapped_files.values():
            mapped_file.mmap.close()
            mapped_file.file.close()
