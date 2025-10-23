
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict, deque

from dnet.utils.logger import logger
from dnet.ring import LayerAssignment, TopologyInfo

Key = Tuple[str, Optional[int], Optional[int], str]  # (node_id, pid, tid, req_id)

@dataclass
class _ActiveSpan:
    """Per-instance active span used for self-time accounting on a call stack."""
    name: str
    t0: int
    child: int = 0


@dataclass
class _SymbolAgg:
    """Aggregated statistics for a single trace symbol (name)."""
    total_ms: float = 0.0
    count: int = 0
    durations: deque = field(default_factory=lambda: deque(maxlen=10000))

    def add(self, self_ms: float) -> None:
        self.total_ms += float(self_ms)
        self.count += 1
        self.durations.append(float(self_ms))

# Sort known frames and compute averages by key  
@dataclass
class RunAggregator:
    sums_by_name: Dict[str, float] = field(default_factory=dict) 
    counts_by_name: Dict[str, int] = field(default_factory=dict)
    last_batch_seq: Dict[str, int] = field(default_factory=dict) 

    stacks: Dict[Key, List[_ActiveSpan]] = field(default_factory=dict)
    drops: int = 0
    # Aggregated stats per symbol (primary source of truth)
    symbols: Dict[str, _SymbolAgg] = field(default_factory=dict)
    # Back-compat mirrors for existing readers (e.g., legacy REPL code)
    durations_by_name: Dict[str, deque] = field(default_factory=dict)

    def _key(self, node_id: str, pid: Optional[int], tid: Optional[int], req_id: Optional[str]) -> Key:
        return (node_id, pid, tid, req_id or "")

    def _push(self, key: Key, f: _ActiveSpan) -> None:
        self.stacks.setdefault(key, []).append(f)

    def _pop(self, key: Key) -> Optional[_ActiveSpan]:
        st = self.stacks.get(key)
        if not st: return None
        return st.pop()

    def _peek(self, key: Key) -> Optional[_ActiveSpan]:
        st = self.stacks.get(key)
        return st[-1] if st else None

    def _accumulate(self, name: str, self_ms: float) -> None:
        sym = self.symbols.get(name)
        if sym is None:
            sym = _SymbolAgg()
            self.symbols[name] = sym
        sym.add(self_ms)

        # FIXME: Remove
        self.sums_by_name[name] = sym.total_ms
        self.counts_by_name[name] = sym.count
        dq = self.durations_by_name.get(name)
        if dq is None:
            dq = deque(maxlen=10000)
            self.durations_by_name[name] = dq
        dq.append(float(self_ms))

    def ingest_event(self, node_id: str, ev: Dict[str, Any]) -> None:
        if not ev.get("name"):
            logger.error(f"Received trace frame without name {ev.get('ts')}")
            return
        # Normalize timestamp to microseconds
        ts_raw = ev.get("ts")
        ts = 0
        try:
            if isinstance(ts_raw, float):
                ts = int(ts_raw * 1_000_000)
            elif isinstance(ts_raw, int):
                ts = ts_raw
            else:
                ts = int(ts_raw or 0)
        except Exception:
            ts = 0
        req_id = ev.get("req_id") or ""
        key = self._key(node_id, ev.get("pid"), ev.get("tid"), req_id)
        if ev.get("type") == "B":
            self._push(key, _ActiveSpan(name=ev.get("name"), t0=ts))
        elif ev.get("type") == "E":
            fr = self._pop(key)
            if not fr: return
            dur_us = max(0, ts - fr.t0)
            self_us = max(0, dur_us - fr.child)
            self_ms = self_us / 1000.0
            self._accumulate(fr.name, self_ms)
            parent = self._peek(key)
            if parent:
                parent.child += dur_us
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
        events = batch.get("events") or []
        logger.debug(f"Enquing trace buffer from {run_id}, {node_id}")
        if not run_id or not node_id: return # Drop batch
        with self._lock:
            agg = self._req.setdefault(run_id, RunAggregator())
            for ev in events:
                agg.ingest_event(node_id, ev)

    def annotate(self, run_id: str, *, mapping: Optional[Dict[str, str]] = None, repeats: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            agg = self._req.get(run_id)
            if not agg:
                return []
            def _stats(xs: List[float]) -> Dict[str, float]:
                if not xs:
                    return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
                n = len(xs)
                srt = sorted(xs)
                def q(p: float) -> float:
                    if n == 1:
                        return srt[0]
                    k = int(round(p * (n - 1)))
                    k = max(0, min(k, n - 1))
                    return srt[k]
                return {
                    "mean": (sum(xs) / n),
                    "p50": q(0.5),
                    "p90": q(0.9),
                    "p99": q(0.99),
                    "min": srt[0],
                    "max": srt[-1],
                }

            rows: List[Dict[str, Any]] = []
            if not mapping:
                for name, sym in agg.symbols.items():
                    total = sym.total_ms
                    samples = list(sym.durations)
                    st = _stats(samples)
                    rows.append({
                        "name": name,
                        "total": total,
                        "max": st["max"],
                        "mean": st["mean"],
                        "p50": st["p50"],
                        "p90": st["p90"],
                        "p99": st["p99"],
                        "samples": len(samples),
                    })
            else:
                sums: Dict[str, float] = {}
                counts: Dict[str, int] = {}
                dists: Dict[str, List[float]] = {}
                for raw, sym in agg.symbols.items():
                    disp = mapping.get(raw, raw)
                    sums[disp] = sums.get(disp, 0.0) + sym.total_ms
                    counts[disp] = counts.get(disp, 0) + sym.count
                    if sym.durations:
                        dists.setdefault(disp, []).extend(sym.durations)
                for name, total in sums.items():
                    samples = dists.get(name, [])
                    st = _stats(samples)
                    rows.append({
                        "name": name,
                        "total": total,
                        "max": st["max"],
                        "mean": st["mean"],
                        "p50": st["p50"],
                        "p90": st["p90"],
                        "p99": st["p99"],
                        "samples": len(samples),
                    })
            rows.sort(key=lambda r: r["total"], reverse=True)
            return rows

    def roots(self, run_id: str, req_id: str) -> List[Dict[str, Any]]:
        # Call-tree storage is disabled to reduce memory; keep API for compatibility.
        return []


# Runtime statistics 
# Use a RunAggregator to get raw frames per request, then transform into _RuntimeStats 

# Track a single request, use multiple for a full benchmark
@dataclass
class _RuntimeStats:
  model: str                # Model name
  tokenizer: str            # Tokenizer name
  run_id: str               # ID of session (for later mapping) 
  nonce: List[str]          # List of serviced requests
  ttft: float               # Time to first token
  itl: float                # Inter-token latency
  requests: int             # Number of requests serviced
  failed: int               # Number of failed requests
  prompt_tokens: int        # Number of prompt tokens per request (req_id: #)
  generated_tokens: int     # Number of generated tokens per request (req_id: #)

  latencys: Dict[List[str, str, str], int]  # Map of latencys: [node0, node1, p50]: 0.0 
  latency_per_layer: Dict[int, float]       # Map of {layer: 0.0} 
  latency_per_shard: Dict[str, float]       # Map of {shard: 0.0}
  total_latency: int        # Total runtime of requests
  throughput: float         # aaa 
  startup_t: float            # Time to start shard (ms)
  layer_assignment_t: float   # Time to layer assignment (ms)

  topo: TopologyInfo = None           # Topology information for this request (keep here since it might change)
  assignment: LayerAssignment = None  # Map of layer to shard IDs


# Process stats + handle per-request data
# NOTE: Hardcodes some high-level trace frame symbols
# TODO: Use a bitmap to track the stages for each req and prevent double-count
class StatsAggregator:
    def __init__(self) -> None:
      self._lock = threading.Lock()

      self._max_inflight_rq = 20 # per node  FIXME: modify from repl

      # Frames are kept in here while in-flight, then remove the frame objects and append to _stats 
      self._workers: Dict[str, Dict[str, Dict[str, Any]]] = {} # Store frames per nonce, per node_id 

      self._nonces = []                                  # Tracked nonces (either in-flight or done)
      self._nonce_round_finish: Dict[str, bool] = {}     # Track in-flight rounds
      self._nonce_prefill: Dict[str, bool] = {}          # Track if this round is prefill
      self._running_stats: Dict[str, _RuntimeStats] = {} # Unfinished stat frames
      self._stats: Dict[str, _RuntimeStats] = {}         # Finished stat frames 
      self._open_frames: Dict[str, Dict[str, Any]]       # We got 'B' event but not 'E' (per nonce)

    # Ingest raw data from tracer
    def add(self, data: Dict[str, Any]) -> None:
      run_id =  data.run_id
      node_id = data.node_id
      events =  data.events or []
      name =    data.name
      if not run_id or not node_id: return # Drop the batch

      with self._lock:
          
          # Ensure we register workers and nodes
          for i, ev in enumerate(events):
              if "nonce" not in ev.attrs: ev.attrs["nonce"] = f"N_{i}"
              nonce = ev.attrs["nonce"] 

              new_frames.append(ev)

              if node_id not in self._workers:
                self._workers[node_id] = {}

              if nonce not in self._workers[node_id]:
                self._workers[node_id][nonce] = {}

              if len(self._workers[node_id]) >= self._max_resident_req: # remove oldest entry
                  del self._workers[self._nonces[0]] 
                  del self._nonces[0]

              self._nonces.push(nonce)

          # Update in-flight events or register new ones
          for e in new_events:
              nonce = e.attrs["nonce"]
              assert nonce is not None, ""

              if not node_id and nonce: return # Drop invalid frames
              stats = self._running_stats[nonce]

              # Register new request
              if e.name == "compute.embedding": 
                  #assert "model" in self._frames, "No model found in trace data."
                  rt_stat = self._running_stats.setdefault(run_id, _RuntimeStats)
                  #rt_stat.model = self._workers[0]["model"][-1].attrs["model"] 
                  #rt_stat.tokenizer = 
                  rt_stat.run_id = run_id
                  rt_stat.nonce = nonce
                  rt_stat.ttft = {}

              if e.name == "network.ingress": 
                  if e.type == "B": self._open_frames[nonce][e.name] = e
                  n_rt = e.attrs["inflight"] + e.attrs["inwait"] 
                  n_rt += self._open_frames[nonce][e.name].t0
                  if self._nonce_prefill[nonce]:
                      stats.ttft += n_rt 
                      continue
                  stats.itl += n_rt

              if f.name == "compute.forward": 

              # Request is finished, construct _RuntimeStats and remove from memory
              if "final" in self._workers[node_id][nonce] and not self._nonce_round_finish[nonce]:
                  self._nonce_round_finish[nonce] = True
                  if not self._nonce_prefill[nonce]: # This is prefill, append ttft

              acc_ttt = 0 # accumulated time to token
              acc_ttt += shard["network.ingress"][-1]
              inflight = shard['network.ingress'][]


    # Return data for total, per req, worker or model (maybe add per layer too?)
    def stats(
      self, 
      req_id: Optional[str], 
      worker: Optional[str], 
      model: Optional[str]
    ):

      if req_id:
        pass

      elif worker:
        pass

      elif model:
        pass

      else: # Return stats of all counters
        pass
        
