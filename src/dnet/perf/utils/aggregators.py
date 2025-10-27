
from __future__ import annotations

import sys
import threading
import statistics 
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict, deque

#from dnet.utils.logger import logger
from dnet.ring.api.api_logging import get_api_logger
from dnet.ring import LayerAssignment, TopologyInfo

logger = get_api_logger()

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
      try:
        run_id = batch.get("run_id")
        node_id = batch.get("node_id")
        events = batch.get("events") or []
        logger.debug(f"Enquing trace buffer from {run_id}, {node_id}")
        if not run_id or not node_id: return # Drop batch
        with self._lock:
            agg = self._req.setdefault(run_id, RunAggregator())
            for ev in events:
                agg.ingest_event(node_id, ev)
      except Exception as e:
        logger.error(f"Trace aggregator enque error: {e}")

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
  model: str = ""              # Model name
  tokenizer: str = ""          # Tokenizer name
  run_id: str = ""             # ID of session (for later mapping) 
  req_id: str = ""             # List of serviced requests
  ttft: float = 0.0            # Time to first token
  itl: List[float] = None      # Inter-token latency per round 
  prompt_tokens: int = -1      # Number of prompt tokens per request (req_id: #)
  generated_tokens: int = -1   # Number of generated tokens per request (req_id: #)
  total_tokens: int = -1       # Total number of tokens processed
  nodes: List[str] = None      # Nodes that participated in computation
  _rounds_t0: List[int] = None # Keep round times for post-processing

  latencies: List[List[str, str, str, int]] = None # List of inter-node latencies: [node0, node1, p50, 0.0]
  latency_per_layer: Dict[int, float] = None       # Map of {layer: 0.0} 
  latency_per_shard: Dict[str, float] = None       # Map of {shard: 0.0}
  total_latency: int = -1                          # Total runtime of requests
  startup_t: float = 0.0                           # Time to start shard (ms)
  layer_assignment_t: float = 0.0                  # Time to layer assignment (ms)

  # Per-worker data
  compute_per_worker: Dict[str, float] = None
  network_per_worker: Dict[str, float] = None
  memory_per_worker: Dict[str, float] = None

  # Network info
  tx_bytes_per_node: Dict[str, int] = None  # Volume of trafic per node 
  rx_bytes_per_node: Dict[str, int] = None

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
      self._frames: Dict[str, Dict[str, Dict[str, Any]]] = {} # Store frames per req_id, per node_id 

      self._req: List[str] = []                          # Tracked requests (in-flight or done)
      self._req_round_finish: Dict[str, bool] = {}       # Track in-flight requests 
      self._req_prefill: Dict[str, bool] = {}            # Track if this request round is prefill
      self._open_frames: Dict[str, Dict[str, Dict[str, Any]]] = {}  

      # Staging environment for events that arrive before 
      # the request.start of the request they belong to
      self._staging: Dict[str, Any] = {}

      # Finished and running requests
      self._running_stats: Dict[str, ReqStats] = {}      # Unfinished stat frames
      self._stats: Dict[str, ReqStats] = {}              # Finished stat frames 

      self.nodes = [] # Keep track of active nodes in the network 

    # Ingest raw data from tracer
    def add(self, data: Dict[str, Any]) -> None:
      events =  data["events"] or []
      if not events: return # Nothing to do 

      node_id = data.get("node_id")
      if not node_id: return # Drop unknown node

      with self._lock:

          # Update in-flight events or register new ones
          for i, e in enumerate(events):
              symbol = e["name"].split(".")

              if e["type"] == 'B':
                req_id = data.get("req_id")
                if not req_id or not node_id: continue 
                self._open_frames[req_id][node_id][e["name"]] = e
                continue
                
              req_id = e["args"].get("req_id")
              if not req_id: 
                #print(f"Dropping {e}")
                continue # Drop anonymous frames 

              if symbol[0] == "request":
                  if symbol[1] == "start": # Start request, setup buffers and ingest staged frames
                      self._req.append(req_id)
                      self._open_frames[req_id] = {}
                      self._req_prefill[req_id] = True 
                      stats = ReqStats(
                          model=e["args"]["model"],
                          tokenizer=e["args"]["tokenizer"], 
                          req_id=req_id,
                          ttft=0.0,
                          itl=[],
                          prompt_tokens=e["args"]["prompt_tokens"],
                          total_tokens=e["args"]["prompt_tokens"],
                          latencies={},
                          latency_per_layer={},
                          latency_per_shard={},
                          assignment=None,
                          compute_per_worker={},
                          network_per_worker={},
                          memory_per_worker={},
                          nodes=[],
                          _rounds_t0=[],
                      )
                      self._running_stats[req_id] = stats 

                      # Process all events in staging
                      if req_id in self._staging:
                        for pe in self._staging[req_id]:
                          node_id = e["args"].get("node_id")
                          if not node_id: continue 
                          self._process_frame(pe, req_id, node_id, stats)

                        del self._staging[req_id]
                      continue

                  elif symbol[1] == "end": # Finish processing request
                      st_obj = self._running_stats[req_id]
                      st_obj.generated_tokens = e["args"]["generated_tokens"]
                      st_obj.total_tokens += e["args"]["generated_tokens"]
                      self._stats[req_id] = st_obj
                      del self._running_stats[req_id]
                      # TODO: Handle latency of transfer back to API
                      continue

                  elif symbol[1] == "round":
                    stats = self._running_stats[req_id]
                    stats._rounds_t0.append(e["args"]["t0"])
                    continue

              if req_id in self._stats: continue # Already finidhed processing request
              if req_id not in self._req:        # If unknown request, stage frames 
                  if req_id not in self._staging:
                    self._staging[req_id] = []
                  self._staging[req_id].append(e)
                  continue

              #node_id = e["args"].get("node_id")
              #if not node_id: return # Drop unknown node

              stats = self._running_stats[req_id]
              self._process_frame(e, req_id, node_id, stats)


    def _process_frame(self, e: Any, req_id: str, node_id: str,  stats: ReqStats):
      symbol = e["name"].split(".")
      if node_id not in self.nodes: self.nodes.append(node_id)
      if node_id not in stats.nodes:
        stats.nodes.append(node_id)
        stats.compute_per_worker[node_id] = 0.0
        stats.network_per_worker[node_id] = 0.0
        stats.memory_per_worker[node_id] = 0.0

      if symbol[0] == "compute":
        if symbol[1] == "forward": 
          pass
          #_cost = lambda e: e["args"]["inwait"] + e["args"]["ms"] 
            # Defer stats compute until after we sort the times (async is kil)
        stats.compute_per_worker[node_id] += e["args"]["ms"] 
        print(f"COMPUTE_PER_WORKER: {e["name"]} : {stats.compute_per_worker}")

      elif symbol[0] == "network":
        if symbol[1] == "rx": # Time in transport, ingress queue and ingress_worker
          _cost = lambda e: e["args"]["inflight"] + e["args"]["inwait"] + e["args"]["ms"]
          #TODO: change shard in metadata
        stats.network_per_worker[node_id] += e["args"]["ms"] 

      elif symbol[0] == "memory":
        stats.memory_per_worker[node_id] += e["args"]["ms"] 
      return

    def _compute_round_stats(self, stats):
      rounds = stats._rounds_t0 
      #rounds.sort()
      assert len(rounds) > 2, "Not enough data."
      stats.ttft = (rounds[1] - rounds[0]) * 1e-6
      stats.itl.append(rounds[1])
      for i in range(1, len(rounds)):
        stats.itl[-1] = (rounds[i] - rounds[i-1]) * 1e-6
        stats.itl.append(rounds[i])
      stats.itl = stats.itl[:-1]

    # Return data for total, per req, worker or model (maybe add per layer too?)
    def stats(
      self, 
      req_id: Optional[str] = None, 
      worker: Optional[str] = None, 
      model: Optional[str] = None
    ):

      # FIXME: Allow manual selection of counters (and push to tracer)
      fields = [ # 0 is native, 1 compound
        (0, "prompt_tokens", ""), 
        (0, "generated_tokens", ""), 
        (0, "total_tokens", ""), 
        (0, -1, ""), # special for empty line 
        (0, "ttft", "ms"), 
        (1, "tokens_per_second", "ms"),
        (1, "inter_token_latency", "ms"),
        (0, -1, ""),
        (1, "estimated_compute", "GFLOPs"),
      ]

      # FIXME: Allow filtering by these
      if req_id: pass
      elif worker: pass
      elif model: pass

      else: # Sort per model, per request (node info only when requested)
        if len(self._stats) < 1:
          print("No tracked stats in memory. Track a request first.\n")
          return
        stats = self._stats[list(self._stats.keys())[-1]]
        self._compute_round_stats(stats)
        #sys.stdout.write(f"\n Loaded model '{stats.model}'.\n")
        sys.stdout.write(f"Performance stats for request '{stats.req_id}':\n\n")
        try:
          for tag, n, unit in fields:
            if tag == 0: # Native counter 
              if n == -1:
                sys.stdout.write("\n")
                continue
              nr = getattr(stats, n)
              if isinstance(nr, int):
                nr_str = f"{nr}"
              elif isinstance(nr, float):
                nr_str = f"{nr:.2f}"
              elif isinstance(nr, str):
                if len(nr) > 20:
                  nr_str = nr[:15] + "..."
                else:
                  nr_str = nr
              sys.stdout.write(f"{nr_str.rjust(20)} {unit.ljust(5)}\t{n}\n")

            # Compound trackers
            elif tag == 1:
              match n:
                case "tokens_per_second":
                  tps = [ 1 / (rt/1000) for rt in stats.itl ]
                  #median = statistics.median(tps)
                  mean = sum(tps) / len(tps)
                  sys.stdout.write(f"{mean:.2f}".rjust(20)+" tok/s".rjust(5)+" \ttokens_per_second")
                  sys.stdout.write(f"\t# {statistics.median(stats.itl)/1000:.3f} s/tok\n")

                case "inter_token_latency":
                  assert len(stats.itl) > 1, "Not enough trace frames"
                  itl = stats.itl[:-1] # FIXME: last element is super big
                  median = statistics.median(itl)
                  p90 = statistics.quantiles(itl, n=100)[89]
                  p99 = statistics.quantiles(itl, n=100)[98]
                  sys.stdout.write(f"{median:.4f}".rjust(20) + " ms".ljust(5) + "\tmean_inter_token_latency (ITL)\n")
                  sys.stdout.write(" "*35 + f"{p90:.3f} (p90),  {p99:.3f} (p99)\n")
                  sys.stdout.write(" "*35 +f"{min(itl):.3f} (min), {max(itl):.3f} (max)\n")

                case "estimated_compute":
                  sys.stdout.write(f"UNKNOWN".rjust(20)+" GFLOPs".ljust(5)+"\testimated_flops\n")

                case _:
                  pass

          for i, n in enumerate(self.nodes):
            comp = stats.compute_per_worker[n]
            net = stats.network_per_worker[n]
            mem = stats.memory_per_worker[n]
            total = comp + net + mem
            sys.stdout.write(f"\n    node{i} [{n}]\n")
            sys.stdout.write(f"{comp:.2f}".rjust(20)+"ms".ljust(5)+f"\tcompute_time   # [{(comp/total)*100:0.2f}%]\n")
            sys.stdout.write(f"{net:.2f}".rjust(20)+"ms".ljust(5)+f"\tnetwork_time   # [{(net/total)*100:0.2f}%]\n")
            sys.stdout.write(f"{mem:.2f}".rjust(20)+"ms".ljust(5)+f"\tmemory_time    # [{(mem/total)*100:0.2f}%]\n")

        except Exception as e:
          logger.error(f"{e}")

        # Per-node information
        sys.stdout.write("\n")
        return
        

