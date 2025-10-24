
from __future__ import annotations

import sys
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
# Use a RunAggregator to get raw frames per request, then transform into ReqStats 

# Track a single request, use multiple for a full benchmark
@dataclass
class ReqStats:
  model: str                # Model name
  tokenizer: str            # Tokenizer name
  run_id: str               # ID of session (for later mapping) 
  nonce: str                # List of serviced requests
  ttft: float               # Time to first token
  itl: List[float]          # Inter-token latency per round 
  prompt_tokens: int        # Number of prompt tokens per request (req_id: #)
  generated_tokens: int     # Number of generated tokens per request (req_id: #)
  total_tokens: int         # Total number of tokens processed

  latencies: List[List[str, str, str, int]] # List of inter-node latencies: [node0, node1, p50, 0.0]
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

      self._max_inflight_req = 20 # per node  FIXME: modify from repl

      # Frames are kept in here while in-flight, then remove the frame objects and append to _stats 
      self._frames: Dict[str, Dict[str, Dict[str, Any]]] = {} # Store frames per nonce, per node_id 

      self._nonces: List[str] = []                       # Tracked nonces (either in-flight or done)
      self._nonce_round_finish: Dict[str, bool] = {}     # Track in-flight rounds
      self._nonce_prefill: Dict[str, bool] = {}          # Track if this round is prefill
      self._running_stats: Dict[str, ReqStats] = {}      # Unfinished stat frames
      self._stats: Dict[str, ReqStats] = {}              # Finished stat frames 
      self._open_frames: Dict[str, Dict[str, Any]] = {}  # We got 'B' event but not 'E' (per nonce)
      self._model_per_run: Dict[str, str] = {}           # Track model per run_id

    # Ingest raw data from tracer
    def add(self, data: Dict[str, Any]) -> None:
      run_id =  data["run_id"] 
      node_id = data["node_id"]
      events =  data["events"] or []

      if not run_id or not node_id: 
        print("Dropped batch")
        return # Drop the batch

      with self._lock:
          
          # Ensure we register workers and nodes
          for i, ev in enumerate(events):
              if "nonce" not in ev["args"]: ev["args"]["nonce"] = f"N_"
              nonce = ev["args"]["nonce"] 

              if node_id not in self._frames:
                self._frames[node_id] = {}

              if nonce not in self._frames[node_id]:
                self._frames[node_id][nonce] = {}

              if len(self._frames[node_id]) >= self._max_inflight_req: # remove oldest entry
                  del self._frames[self._nonces[0]] 
                  del self._nonces[0]

              if nonce not in self._nonces:
                  self._nonces.append(nonce)

          # Update in-flight events or register new ones
          for e in events:
              nonce = e["args"]["nonce"]
              assert nonce is not None, ""


              if not node_id or not nonce: return # Drop invalid frames

              if e["name"] == "chat.request.start":
                  print(e["args"])
                  self._open_frames[nonce] = {}
                  self._nonce_prefill[nonce] = True 
                  self._running_stats[nonce] = ReqStats(
                      model=e["args"]["model"],
                      tokenizer=e["args"]["tokenizer"], 
                      run_id=run_id,
                      nonce=nonce,
                      ttft=e["args"]["t0"],  # set to initial timestamp then compute
                      itl=[0.0],
                      generated_tokens=0,
                      prompt_tokens=e["args"]["prompt_tokens"],
                      total_tokens=e["args"]["prompt_tokens"],
                      latencies={},
                      latency_per_layer={},
                      latency_per_shard={},
                      total_latency=0.0,
                      assignment=None,
                      topo=None,
                      layer_assignment_t=None,
                      throughput=0.0,
                      startup_t=0.0,
                  )


              if e["name"] == "embedding": # Register new request
                  pass

              # FIXME: We might receive other frames then "embed" from shards
              #        so we need to handle the creation of this better
              if nonce not in self._running_stats:
                continue

              stats = self._running_stats[nonce]

              if e["name"] == "network.rx": # Time in transport, ingress queue and ingress_worker
                print(f"\n{e["name"]}\n{e["args"]["inflight"]}\n{e["args"]["inwait"]}\n{e["args"]["ms"]}")
                _cost = lambda e: e["args"]["inflight"] + e["args"]["inwait"] + e["args"]["ms"]
                self._handle_frame(e, nonce, stats, _cost)
                #TODO: change shard in metadata

              if e["name"] == "compute.forward": 
                print(f"\n{e["name"]}\n{e["args"]["inwait"]}\n{e["args"]["ms"]}")
                _cost = lambda e: e["args"]["inwait"] + e["args"]["ms"] # compute queue + execution
                self._handle_frame(e, nonce, stats, _cost)
                self._nonce_round_finish[nonce] = False

                # End a cycle on compute done (inter-node queue wait is computed in next)
                if self._nonce_prefill[nonce]:
                  stats.ttft = e["args"]["t0"] - stats.ttft
                else:
                  stats.itl[-1] = e["args"]["t0"] - stats.itl[-1]
                  stats.itl.append(e["args"]["t0"])

              if e["name"] == "chat.request.end":
                if self._nonce_round_finish[nonce]:
                  self._nonce_round_finish[nonce] = True 
                  pass
                self._nonce_round_finish[nonce] = True
                st_obj = self._running_stats[nonce]
                st_obj.generated_tokens = e["args"]["generated_tokens"]
                st_obj.total_tokens += e["args"]["generated_tokens"]
                self._stats[nonce] = st_obj
                del self._running_stats[nonce]
                #del self._frames[node_id][nonce]
                # TODO: Handle latency of transfer back to API

              if e["name"] == "network.tx.send":
                _cost = lambda e: e["args"]["inwait"] + e["args"]["ms"] # tx queue + sendoff
                self._handle_frame(e, nonce, stats, _cost)

    # Handle cost aggregation of frames
    def _handle_frame(self, e: Any, nonce, stats: ReqStats, _cost_fnc: Any):
      try:
        if e["type"] == 'B': 
          self._open_frames[nonce][e["name"]] = e
          return
        elif e["type"] == 'E':
          n_rt = _cost_fnc(e) # Custom cost function for each farme 
          if self._nonce_prefill[nonce]:
            stats.ttft += n_rt
          else:
            stats.itl[-1] += n_rt
          if e["name"] in self._open_frames[nonce]:
            del self._open_frames[nonce][e["name"]]
      except Exception as ex:
        print(f"{ex}")

    # Return data for total, per req, worker or model (maybe add per layer too?)
    def stats(
      self, 
      req_id: Optional[str] = None, 
      worker: Optional[str] = None, 
      model: Optional[str] = None
    ):

      fields = [ # 0 is native, 1 compound
        (0, "prompt_tokens", ""), 
        (0, "generated_tokens", ""), 
        (0, "total_tokens", ""), 
        (0, -1, ""), # special for empty line 
        (0, "ttft", "ms"), 
        (1, "tokens_per_second", "ms"),
        (1, "inter_token_latency", "ms"),
        (0, "throughput", "ms"), 
        (0, -1, ""),
        (1, "workers", "ms"), 
        (1, "estimated_compute", "GFLOPs")
      ]

      if req_id:
        pass

      elif worker:
        pass

      elif model:
        pass

      else: # Sort per model, per request (node info only when requested)
        if len(self._stats) < 1:
          print("No tracked stats in memory. Track a request first.\n")
          return
        stats = self._stats[list(self._stats.keys())[-1]]
        sys.stdout.write(f"\n Performance counters stats for model '{stats.model}':\n\n")
        for tag, n, unit in fields:
          if tag == 0: # Native counter 
            if n == -1:
              sys.stdout.write("\n")
              continue
            nr = getattr(stats, n)
            if isinstance(nr, int):
              nr_str = f"{nr}"
            elif isinstance(nr, float):
              nr_str = f"{nr:15.5}"
            elif isinstance(nr, str):
              if len(nr) > 20:
                nr_str = nr[:15] + "..."
              else:
                nr_str = nr
          elif tag == 1:
            match n:
              case "tokens_per_second":
              case "inter_token_latency":
              case _:
                pass
          sys.stdout.write(f"{nr_str.rjust(20)} {unit.ljust(4)}\t{n}\n")
        sys.stdout.write("\n\n")
        return
        

