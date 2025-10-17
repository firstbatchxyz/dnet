from __future__ import annotations

import math
from typing import Optional, Dict, Tuple

import numpy as np
import mlx.core as mx

from .ops import (
    _compute_norms_fp32,
    _compute_norms_quant_int8,
    _select_k_smallest_indices,
)
from .kernels import (
    k_gather_cols,
    k_scatter_from_compact,
    k_dequant_scatter_q8,
)
from .utils import _u32, infer_groups_and_gsize, reshape_rg


def compress_tensor_to_protobuf_data(
    tensor: mx.array,
    compression_percentage: float = 90.0,
    *,
    quant: Optional[Dict[str, mx.array]] = None,
) -> Tuple[bytes, list, str]:
    """Serialize tensor with true sparse payload.

    Unquantized (sparse_v1): send D-bit column mask + kept values (R x K) in fp16 for fp32/bf16 inputs.
    Quantized 8-bit (qsparse8_v1): send D-bit column mask + group mask + kept uint8 codes + compact per-row kept-group params (fp16).
    """
    if not isinstance(tensor, mx.array):
        raise TypeError("Input must be an MLX array.")
    if not 0.0 <= compression_percentage <= 100.0:
        raise ValueError("compression_percentage must be 0..100.")

    if tensor.size == 0 or tensor.shape[-1] == 0:
        comp_dtype = (
            "float16"
            if (quant is not None or tensor.dtype in (mx.bfloat16, mx.float32))
            else str(tensor.dtype).replace("mlx.core.", "")
        )
        return (
            b"",
            list(tensor.shape),
            f"{comp_dtype}|{compression_percentage}|q={(1 if quant is not None else 0)}|orig={str(tensor.dtype)}",
        )

    D = tensor.shape[-1]
    R = tensor.size // D
    k = min(math.ceil((compression_percentage / 100.0) * D), D)

    force_fp16 = (quant is not None) or (tensor.dtype in (mx.bfloat16, mx.float32))
    comp_dtype = "float16" if force_fp16 else str(tensor.dtype).replace("mlx.core.", "")
    qflag = 1 if quant is not None else 0

    if k == 0:
        arr = tensor.astype(mx.float16) if comp_dtype == "float16" else tensor
        buf = np.asarray(arr).tobytes()
        meta = (
            f"{comp_dtype}|{compression_percentage}|q={qflag}|orig={str(tensor.dtype)}"
        )
        return buf, list(tensor.shape), meta

    # Norms + selection
    if quant is None:
        x32 = tensor.astype(mx.float32).reshape(R, D)
        norms = _compute_norms_fp32(x32)
    else:
        q = tensor.reshape(R, D)
        scales = quant["scales"]
        biases = quant.get("biases", None)
        gsz_in = int(quant.get("group_size", 64))
        norms = _compute_norms_quant_int8(q, scales, biases, gsz_in)
    idx = _select_k_smallest_indices(norms, k)

    drop_idx_np = np.asarray(idx, dtype=np.int64)
    keep_bits = np.ones((D,), dtype=bool)
    keep_bits[drop_idx_np] = False
    mask_bytes = np.packbits(keep_bits.astype(np.uint8), bitorder="little").tobytes()
    keep_idx_np = np.nonzero(keep_bits)[0].astype(np.uint32)
    K = int(keep_idx_np.size)
    keep_idx_mx = mx.array(keep_idx_np, dtype=mx.uint32)

    if quant is None:
        x2d = tensor.reshape(R, D)
        tgx = 512 if (R * K) >= (1 << 20) else 256
        kept_gpu = k_gather_cols(
            inputs=[x2d, keep_idx_mx, _u32([R, D, K])],
            template=[("T", x2d.dtype)],
            grid=(R * K, 1, 1),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(R, K)],
            output_dtypes=[x2d.dtype],
        )[0]
        if comp_dtype == "float16":
            kept_gpu = kept_gpu.astype(mx.float16)
            np_dtype = np.float16
        elif comp_dtype == "float32":
            np_dtype = np.float32
        else:
            np_dtype = None
        kept_cpu = np.asarray(kept_gpu)
        if np_dtype is not None:
            kept_cpu = kept_cpu.astype(np_dtype, copy=False)
        payload = kept_cpu.tobytes()
        buf = mask_bytes + payload
        meta = f"{comp_dtype}|{compression_percentage}|q={qflag}|orig={str(tensor.dtype)}|fmt=sparse_v1"
        return buf, list(tensor.shape), meta

    bits = int(quant.get("bits", 8))
    gsz = int(quant.get("group_size", 64))
    if bits == 8:
        q2d = tensor.reshape(R, D)
        if q2d.dtype != mx.uint8:
            q2d = q2d.astype(mx.uint8)
        tgx = 512 if (R * K) >= (1 << 20) else 256
        kept_codes_gpu = k_gather_cols(
            inputs=[q2d, keep_idx_mx, _u32([R, D, K])],
            template=[("T", q2d.dtype)],
            grid=(R * K, 1, 1),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(R, K)],
            output_dtypes=[mx.uint8],
        )[0]
        kept_codes = np.asarray(kept_codes_gpu)

        G, gsz_eff = infer_groups_and_gsize(
            D, gsz, quant["scales"], quant.get("biases", None)
        )
        g_idx = np.arange(D, dtype=np.int64) // gsz_eff
        g_keep = np.zeros((G,), dtype=bool)
        g_keep[np.unique(g_idx[keep_bits])] = True
        gmask_bytes = np.packbits(g_keep.astype(np.uint8), bitorder="little").tobytes()

        s_rg = reshape_rg(quant["scales"], R, G).astype(mx.float32)
        if quant.get("biases", None) is None:
            b_rg = None
            has_b = 0
        else:
            b_rg = reshape_rg(quant["biases"], R, G).astype(mx.float32)
            has_b = 1

        g_keep_idx_np = np.nonzero(g_keep)[0].astype(np.uint32)
        Gk = int(g_keep_idx_np.size)
        g_keep_idx_mx = mx.array(g_keep_idx_np, dtype=mx.uint32)
        tgx_g = 512 if (R * Gk) >= (1 << 20) else 256
        s_kept_gpu = k_gather_cols(
            inputs=[s_rg, g_keep_idx_mx, _u32([R, G, Gk])],
            template=[("T", s_rg.dtype)],
            grid=(R * Gk, 1, 1),
            threadgroup=(tgx_g, 1, 1),
            output_shapes=[(R, Gk)],
            output_dtypes=[mx.float32],
        )[0].astype(mx.float16)
        s_kept = np.asarray(s_kept_gpu)
        parts = [mask_bytes, gmask_bytes, kept_codes.tobytes(), s_kept.tobytes()]
        if has_b:
            b_kept_gpu = k_gather_cols(
                inputs=[b_rg, g_keep_idx_mx, _u32([R, G, Gk])],
                template=[("T", b_rg.dtype)],
                grid=(R * Gk, 1, 1),
                threadgroup=(tgx_g, 1, 1),
                output_shapes=[(R, Gk)],
                output_dtypes=[mx.float32],
            )[0].astype(mx.float16)
            parts.append(np.asarray(b_kept_gpu).tobytes())
        buf = b"".join(parts)
        meta = f"uint8|{compression_percentage}|q=1|orig={str(tensor.dtype)}|fmt=qsparse8_v1|b=8|gs={gsz_eff}|hb={has_b}|rd=float16"
        return buf, list(tensor.shape), meta

    deq = mx.dequantize(
        tensor.reshape(R, D),
        quant["scales"],
        quant.get("biases", None),
        group_size=gsz,
        bits=bits,
        mode=str(quant.get("mode", "affine")),
    ).astype(mx.float16 if comp_dtype == "float16" else mx.float32)
    tgx = 512 if (R * K) >= (1 << 20) else 256
    kept_gpu = k_gather_cols(
        inputs=[deq, keep_idx_mx, _u32([R, D, K])],
        template=[("T", deq.dtype)],
        grid=(R * K, 1, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(R, K)],
        output_dtypes=[deq.dtype],
    )[0]
    payload = np.asarray(kept_gpu).tobytes()
    buf = mask_bytes + payload
    meta = f"{comp_dtype}|{compression_percentage}|q={qflag}|orig={str(tensor.dtype)}|fmt=sparse_v1"
    return buf, list(tensor.shape), meta


def decompress_tensor_from_protobuf_data(
    tensor_data: bytes, shape: list, dtype_with_metadata: str
) -> mx.array:
    parts = (dtype_with_metadata or "").split("|")
    comp_dtype = parts[0] if parts and parts[0] else "float32"
    orig = "mlx.core.float32"
    fmt = None
    for p in parts[1:]:
        if p.startswith("orig="):
            orig = p.split("=", 1)[1]
        if p.startswith("fmt="):
            fmt = p.split("=", 1)[1]
    np_map = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "int64": np.int64,
        "bfloat16": np.float16,
    }
    np_dtype = np_map.get(comp_dtype, np.float32)

    shp = tuple(shape)
    if fmt == "sparse_v1":
        if not shp or shp[-1] == 0:
            return mx.array(
                np.zeros(shp, dtype=np_dtype),
                dtype={
                    "bfloat16": mx.bfloat16,
                    "float32": mx.float32,
                    "float16": mx.float16,
                }.get(comp_dtype, mx.float32),
            )
        D = int(shp[-1])
        R = int(np.prod(shp) // D)
        mask_bytes = (D + 7) // 8
        if len(tensor_data) < mask_bytes:
            raise ValueError("Sparse buffer too small for mask")
        mask_buf = np.frombuffer(tensor_data[:mask_bytes], dtype=np.uint8)
        mask_bits = np.unpackbits(mask_buf, bitorder="little")[:D].astype(bool)
        K = int(mask_bits.sum())
        expected = K * max(1, R) * np.dtype(np_dtype).itemsize
        if len(tensor_data) != mask_bytes + expected:
            raise ValueError("Sparse buffer size mismatch")
        pos = -np.ones((D,), dtype=np.int32)
        pos[np.nonzero(mask_bits)[0]] = np.arange(K, dtype=np.int32)
        pos_mx = mx.array(pos, dtype=mx.int32)
        vals = np.frombuffer(tensor_data[mask_bytes:], dtype=np_dtype).reshape(R, K)
        vals_mx = mx.array(
            vals,
            dtype={
                "bfloat16": mx.bfloat16,
                "float32": mx.float32,
                "float16": mx.float16,
            }.get(comp_dtype, mx.float32),
        )
        tgx = 512 if (R * D) >= (1 << 20) else 256
        od = orig.replace("mlx.core.", "")
        mx_dtype = {
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
            "float16": mx.float16,
            "int32": mx.int32,
            "int64": mx.int64,
        }.get(od, mx.float32)
        full = k_scatter_from_compact(
            inputs=[vals_mx.reshape(-1), pos_mx, _u32([R, D, K])],
            template=[("T", vals_mx.dtype)],
            grid=(R * D, 1, 1),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(R, D)],
            output_dtypes=[mx_dtype],
        )[0]
        return full.reshape(shp)

    if fmt == "qsparse8_v1":
        b = 8
        gsz = None
        has_b = 0
        rd = "float16"
        for p in parts[1:]:
            if p.startswith("b="):
                try:
                    b = int(p.split("=", 1)[1])
                except Exception:
                    b = 8
            if p.startswith("gs="):
                try:
                    gsz = int(p.split("=", 1)[1])
                except Exception:
                    gsz = None
            if p.startswith("hb="):
                try:
                    has_b = int(p.split("=", 1)[1])
                except Exception:
                    has_b = 0
            if p.startswith("rd="):
                rd = p.split("=", 1)[1]
        if b != 8:
            raise ValueError("qsparse8_v1 expects b=8")
        if gsz is None:
            raise ValueError("qsparse8_v1 missing group size")
        shp = tuple(shape)
        if not shp or shp[-1] == 0:
            return mx.zeros(shp, dtype=mx.float16 if rd == "float16" else mx.float32)
        D = int(shp[-1])
        R = int(np.prod(shp) // D)
        G = (D + gsz - 1) // gsz
        col_mask_bytes = (D + 7) // 8
        if len(tensor_data) < col_mask_bytes:
            raise ValueError("qsparse8_v1 buffer too small for column mask")
        col_mask = np.unpackbits(
            np.frombuffer(tensor_data[:col_mask_bytes], dtype=np.uint8),
            bitorder="little",
        )[:D].astype(bool)
        off = col_mask_bytes
        grp_mask_bytes = (G + 7) // 8
        if len(tensor_data) < off + grp_mask_bytes:
            raise ValueError("qsparse8_v1 buffer too small for group mask")
        grp_mask = np.unpackbits(
            np.frombuffer(tensor_data[off : off + grp_mask_bytes], dtype=np.uint8),
            bitorder="little",
        )[:G].astype(bool)
        off += grp_mask_bytes
        K = int(col_mask.sum())
        Gk = int(grp_mask.sum())
        codes_bytes = R * K
        if len(tensor_data) < off + codes_bytes:
            raise ValueError("qsparse8_v1 buffer too small for codes")
        kept_codes = np.frombuffer(
            tensor_data[off : off + codes_bytes], dtype=np.uint8
        ).reshape(R, K)
        off += codes_bytes
        s_count = R * Gk
        s_bytes = s_count * np.dtype(np.float16).itemsize
        if len(tensor_data) < off + s_bytes:
            raise ValueError("qsparse8_v1 buffer too small for scales")
        s_kept = (
            np.frombuffer(tensor_data[off : off + s_bytes], dtype=np.float16)
            .astype(np.float32)
            .reshape(R, Gk)
        )
        off += s_bytes
        if has_b:
            b_count = R * Gk
            b_bytes = b_count * np.dtype(np.float16).itemsize
            if len(tensor_data) < off + b_bytes:
                raise ValueError("qsparse8_v1 buffer too small for biases")
            b_kept = (
                np.frombuffer(tensor_data[off : off + b_bytes], dtype=np.float16)
                .astype(np.float32)
                .reshape(R, Gk)
            )
            off += b_bytes
        else:
            b_kept = None
        if off != len(tensor_data):
            raise ValueError("qsparse8_v1 buffer has extra bytes")

        pos = -np.ones((D,), dtype=np.int32)
        pos[np.nonzero(col_mask)[0]] = np.arange(K, dtype=np.int32)
        pos_mx = mx.array(pos, dtype=mx.int32)
        g_keep_idx = np.nonzero(grp_mask)[0]
        inv_map = -np.ones(G, dtype=np.int32)
        inv_map[g_keep_idx] = np.arange(Gk, dtype=np.int32)
        inv_map_mx = mx.array(inv_map, dtype=mx.int32)
        codes_mx = mx.array(kept_codes, dtype=mx.uint8)
        s_kept_mx = mx.array(s_kept, dtype=mx.float32)
        b_kept_mx = (
            mx.array(b_kept, dtype=mx.float32)
            if b_kept is not None
            else mx.array(np.zeros((R, Gk), dtype=np.float32))
        )
        tgx = 512 if (R * D) >= (1 << 20) else 256
        out_dtype = mx.float16 if rd == "float16" else mx.float32
        full = k_dequant_scatter_q8(
            inputs=[
                codes_mx.reshape(-1),
                s_kept_mx.reshape(-1),
                b_kept_mx.reshape(-1),
                pos_mx,
                inv_map_mx,
                _u32([R, D, K, G, Gk, gsz, has_b]),
            ],
            template=[("T", mx.uint8)],
            grid=(R * D, 1, 1),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(R, D)],
            output_dtypes=[out_dtype],
        )[0]
        return full.reshape(shp)

    expected = int(np.prod(shp))
    if len(tensor_data) != expected * np.dtype(np_dtype).itemsize:
        if len(tensor_data) == expected * np.dtype(np.float32).itemsize:
            np_dtype = np.float32
        else:
            raise ValueError("Buffer size mismatch")
    arr = np.frombuffer(tensor_data, dtype=np_dtype).reshape(shp)
    od = orig.replace("mlx.core.", "")
    mx_dtype = {
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
        "float16": mx.float16,
        "int32": mx.int32,
        "int64": mx.int64,
    }.get(od, mx.float32)
    return mx.array(arr, dtype=mx_dtype)
