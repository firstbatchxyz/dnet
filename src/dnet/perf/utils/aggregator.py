
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict, deque

from dnet.utils.logger import logger

Key = Tuple[str, Optional[int], Optional[int], str]  # (node_id, pid, tid, req_id)

@dataclass
class _OpenFrame:
    name: str
    t0: int
    child: int = 0
    children: List[Dict[str, Any]] = field(default_factory=list)

# Sort known frames and compute averages by key  
@dataclass
class RunAggregator:
    sums_by_name: Dict[str, float] = field(default_factory=dict) 
    counts_by_name: Dict[str, int] = field(default_factory=dict)
    last_batch_seq: Dict[str, int] = field(default_factory=dict) 

    stacks: Dict[Key, List[_OpenFrame]] = field(default_factory=dict)
    drops: int = 0
    roots_by_req: DefaultDict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))

    def _key(self, node_id: str, pid: Optional[int], tid: Optional[int], req_id: Optional[str]) -> Key:
        return (node_id, pid, tid, req_id or "")

    def _push(self, key: Key, f: _OpenFrame) -> None:
        self.stacks.setdefault(key, []).append(f)

    def _pop(self, key: Key) -> Optional[_OpenFrame]:
        st = self.stacks.get(key)
        if not st: return None
        return st.pop()

    def _peek(self, key: Key) -> Optional[_OpenFrame]:
        st = self.stacks.get(key)
        return st[-1] if st else None

    def _acc_annotate(self, name: str, self_ms: float) -> None:
        self.sums_by_name[name] = self.sums_by_name.get(name, 0.0) + self_ms
        self.counts_by_name[name] = self.counts_by_name.get(name, 0) + 1

    def ingest_event(self, node_id: str, ev: Dict[str, Any]) -> None:
        if not ev.get("name"):
            logger.error(f"Received trace frame without name {ev.get('ts')}")
            return
        # Normalize ts to microseconds (accept float seconds or int microseconds)
        ts_raw = ev.get("ts")
        ts_us = 0
        try:
            if isinstance(ts_raw, float):
                ts_us = int(ts_raw * 1_000_000)
            elif isinstance(ts_raw, int):
                ts_us = ts_raw
            else:
                ts_us = int(ts_raw or 0)
        except Exception:
            ts_us = 0
        req_id = ev.get("req_id") or ""
        key = self._key(node_id, ev.get("pid"), ev.get("tid"), req_id)
        if ev.get("type") == "B":
            self._push(key, _OpenFrame(name=ev.get("name"), t0=ts_us))
        elif ev.get("type") == "E":
            fr = self._pop(key)
            if not fr: return
            dur_us = max(0, ts_us - fr.t0)
            self_us = max(0, dur_us - fr.child)
            self_ms = self_us / 1000.0
            self._acc_annotate(fr.name, self_ms)
            parent = self._peek(key)
            completed = {
                "name": fr.name,
                "ts": fr.t0,
                "dur_ms": dur_us / 1000.0,
                "self_ms": self_ms,
                "children": fr.children,
                "pid": ev.get("pid"),
                "tid": ev.get("tid"),
                "req_id": req_id,
                "node_id": node_id,
            }
            if parent:
                parent.child += dur_us
                parent.children.append(completed)
            else:
                self.roots_by_req[req_id or ""].append(completed)
        else:
            # TODO :Process other events
            pass


class TraceAggregator:
    def __init__(self) -> None:
        self._req: Dict[str, RunAggregator] = {}
        self._lock = threading.Lock()

    def enqueue(self, batch: Dict[str, Any]) -> None:
        run_id = batch.get("run_id")
        node_id = batch.get("node_id")
        if not run_id or not node_id:
            return
        events = batch.get("events") or []
        batch_seq = int(batch.get("batch_seq") or 0)
        with self._lock:
            agg = self._req.setdefault(run_id, RunAggregator())
            last = agg.last_batch_seq.get(node_id)
            if (last is not None) and (batch_seq != last + 1):
                agg.drops += abs(batch_seq - (last + 1))
            agg.last_batch_seq[node_id] = batch_seq
            for ev in events:
                try:
                    agg.ingest_event(node_id, ev)
                except Exception:
                    continue

    def annotate(self, run_id: str, *, mapping: Optional[Dict[str, str]] = None, repeats: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            agg = self._req.get(run_id)
            if not agg:
                return []
            if not mapping:
                rows = [
                    {"name": k, "self_ms": v, "total_ms": v, "count": repeats or agg.counts_by_name.get(k, 0), "max_ms": None}
                    for k, v in agg.sums_by_name.items()
                ]
            else:
                sums: Dict[str, float] = {}
                counts: Dict[str, int] = {}
                for raw, val in agg.sums_by_name.items():
                    disp = mapping.get(raw, raw)
                    sums[disp] = sums.get(disp, 0.0) + val
                    counts[disp] = counts.get(disp, 0) + agg.counts_by_name.get(raw, 0)
                rows = [
                    {"name": k, "self_ms": v, "total_ms": v, "count": repeats or counts.get(k, 0), "max_ms": None}
                    for k, v in sums.items()
                ]
            rows.sort(key=lambda r: r["self_ms"], reverse=True)
            return rows

    def roots(self, run_id: str, req_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            agg = self._req.get(run_id)
            if not agg:
                return []
            return list(agg.roots_by_req.get(req_id or "", []))
