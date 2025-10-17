from __future__ import annotations

from typing import Optional, Tuple
import mlx.core as mx


def _u32(vals):
    return mx.array(vals, dtype=mx.uint32)


def infer_groups_and_gsize(
    D: int, group_size: int, scales: Optional[mx.array], biases: Optional[mx.array]
) -> Tuple[int, int]:
    """Infer (G, gsize) from provided params safely.
    Prefer dimensions implied by scales/biases if available; fallback to provided group_size.
    """

    def _g_from(arr) -> Optional[int]:
        if arr is None:
            return None
        if not hasattr(arr, "shape"):
            return None
        shp = tuple(arr.shape)
        if len(shp) == 1:
            return shp[0]
        if len(shp) == 2:
            r1, g1 = shp
            if g1 > 1 and (r1 == 1 or r1 > 0):
                return g1
            if r1 > 1 and g1 == 1:
                return r1
        return None

    G_hint = _g_from(scales) or _g_from(biases)
    if G_hint is None or G_hint <= 0:
        G = (D + group_size - 1) // group_size
    else:
        G = int(G_hint)
    gsize = max(1, (D + G - 1) // G)
    return G, gsize


def reshape_rg(arr: Optional[mx.array], R: int, G: int) -> mx.array:
    """Return float32 array of shape (R, G) from various input shapes.
    Accepts: (R, G), (1, G), (R, 1), (G,), (R,), scalar, or None (-> zeros).
    """
    if arr is None:
        return mx.zeros((R, G), dtype=mx.float32)
    a = arr if isinstance(arr, mx.array) else mx.array(arr)
    a = a.astype(mx.float32)

    if getattr(a, "ndim", 0) == 2 and a.shape == (R, G):
        return a

    if a.ndim == 0:
        return mx.full((R, G), float(a.item()), dtype=mx.float32)
    if a.ndim == 1:
        if a.shape[0] == G:
            return mx.broadcast_to(a.reshape(1, G), (R, G))
        if a.shape[0] == R:
            return mx.broadcast_to(a.reshape(R, 1), (R, G))
    if a.ndim == 2:
        if a.shape[0] == 1 and a.shape[1] == G:
            return mx.broadcast_to(a, (R, G))
        if a.shape[0] == R and a.shape[1] == 1:
            return mx.broadcast_to(a, (R, G))

    # Fallback to broadcast attempts
    try:
        if a.ndim == 1:
            if a.shape[0] == G:
                a2 = a.reshape(1, G)
            elif a.shape[0] == R:
                a2 = a.reshape(R, 1)
            else:
                a2 = a.reshape(1, -1)
            return mx.broadcast_to(a2, (R, G))
        if a.ndim == 0:
            return mx.broadcast_to(a.reshape(1, 1), (R, G))
        return mx.broadcast_to(a, (R, G))
    except Exception:
        pass
    raise ValueError(
        f"Cannot reshape/broadcast params of shape {tuple(a.shape)} to (R,G)=({R},{G})"
    )
