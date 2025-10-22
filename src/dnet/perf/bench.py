
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from dnet.perf.trace import Tracer


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = int(round(q * (len(ys) - 1)))
    k = max(0, min(k, len(ys) - 1))
    return ys[k]


def collect_stats(times_ms: List[float], *, bytes_total: float = 0.0, tokens_total: float = 0.0) -> Dict[str, Any]:
    if not times_ms:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "samples": 0,
            "mb_s": 0.0,
            "tok_s": 0.0,
        }
    total_ms = sum(times_ms)
    mean = total_ms / len(times_ms)
    std = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    total_s = max(total_ms / 1000.0, 1e-12)
    return {
        "mean": mean,
        "std": std,
        "min": min(times_ms),
        "p50": _percentile(times_ms, 0.5),
        "p90": _percentile(times_ms, 0.9),
        "p99": _percentile(times_ms, 0.99),
        "max": max(times_ms),
        "samples": len(times_ms),
        "mb_per_s": (bytes_total / 1_000_000.0) / total_s if bytes_total else 0.0,
        "tokens_per_s": (tokens_total / total_s) if tokens_total else 0.0,
    }


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


@dataclass
class BenchCounters:
    values: Dict[str, float] = field(default_factory=dict)

    def add_time(self, key: str, dt_ms: float) -> None:
        self.values[key] = self.values.get(key, 0.0) + float(dt_ms)

    def add_bytes(self, *, direction: str, n: int) -> None:
        k = "bytes_in" if direction == "in" else "bytes_out"
        self.values[k] = self.values.get(k, 0.0) + float(n)

    def inc(self, key: str, delta: float = 1.0) -> None:
        self.values[key] = self.values.get(key, 0.0) + float(delta)

    def snapshot(self, *, run_id: str, node: str, role: str = "shard") -> Dict[str, Any]:
        snap = {
            "run_id": run_id,
            "node": node,
            "role": role,
            "counters": dict(self.values),
        }
        return snap


class TimedSpan:
    __slots__ = ("_tracer", "_name", "_attrs", "_t0", "_frame", "_counters", "_counter_key")

    def __init__(
        self,
        tracer: Optional[Tracer],
        name: str,
        counters: Optional[BenchCounters] = None,
        counter_key: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._attrs = attrs or {}
        self._t0 = 0.0
        self._frame = None
        self._counters = counters
        self._counter_key = counter_key

    def __enter__(self):
        self._t0 = time.perf_counter()
        if self._tracer is not None:
            self._frame = self._tracer.frame("bench", self._name, self._attrs)
            self._frame.__enter__()
        return self

    def __exit__(self, ex_type, ex, tb) -> bool:
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        if self._frame is not None:
            try:
                self._frame.__exit__(ex_type, ex, tb)
            except Exception:
                pass
        if self._counters is not None and self._counter_key:
            self._counters.add_time(self._counter_key, dt_ms)
        return False


def aggregate_annotate(
    snapshots: Iterable[Dict[str, Any]],
    *,
    mapping: Optional[Dict[str, str]] = None,
    repeats: int = 0,
) -> List[Dict[str, Any]]:

    sums: Dict[str, float] = {}
    for snap in snapshots:
        ctr = snap.get("counters") if isinstance(snap, dict) else None
        if not isinstance(ctr, dict):
            continue
        for k, v in ctr.items():
            name = mapping.get(k, k) if mapping else k
            try:
                sums[name] = sums.get(name, 0.0) + float(v)
            except Exception:
                continue

    rows = [ {"name": name, "self_ms": val, "total_ms": val, "count": repeats or 0, "max_ms": None}
             for name, val in sums.items() if val > 0.0]
    rows.sort(key=lambda r: r["self_ms"], reverse=True)
    return rows

