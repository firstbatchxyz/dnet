
from __future__ import annotations

import os
import io
import sys
import time
import json
import pstats
import cProfile
import threading
import queue

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from contextlib import contextmanager

import httpx

from dnet.utils.logger import logger

@dataclass
class TraceConfig:
    file: str = "logs/dnet-trace.jsonl"
    streaming: bool = True
    include_prefixes: Tuple[str, ...] = ("src/dnet/",)
    include_c_calls: bool = False
    budget: int = 0  # 0 means unlimited
    enabled: bool = True
    node_id: Optional[str] = None
    record_pid_tid: bool = True
    aggregate: bool = False
    aggregate_url: Optional[str] = None
    agg_max_events: int = 300 

class _NoopFrame:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def event(self, *a, **k):
        pass
    def set(self, *a, **k):
        pass

class _Frame:
    __slots__ = ("t", "name", "attrs", "_t0")
    def __init__(self, tracer: "Tracer", name: str, attrs: Optional[Dict[str, Any]]):
        self.t = tracer
        self.name = name
        self.attrs = dict(attrs or {})
        self._t0 = 0.0
    def __enter__(self):
        self._t0 = time.time_ns() # cross-node timekeeping
        self.t._emit({"type": "B", "name": self.name, "args": dict(self.attrs)})
        return self
    def __exit__(self, ex_type, ex, tb):
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self.attrs.update({"ms": round(dt_ms, 3), "exc": bool(ex)})
        self.t._emit({"type": "E", "name": self.name, "args": self.attrs})
        return False
    def event(self, name: str, **attrs):
        out = dict(attrs or {})
        out.setdefault("t_rel_ms", (time.perf_counter() - self._t0) * 1000.0)
        self.t._emit({"type": "I", "name": f"{self.name}.{name}", "args": out})
    def set(self, key: str, val: Any):
        self.attrs[key] = val

class Tracer:
    def __init__(self, config: TraceConfig):
        self.config = config
        self._lock = threading.Lock()
        self._fh: Optional[io.TextIOBase] = None
        self._events: List[Dict[str, Any]] = []
        self._req_id: str = None
        self._active = False

        self._agg_enabled: bool = False
        self._agg_max_events: int = int(self.config.agg_max_events or 1000)
        self._agg_q: queue.Queue[dict] = queue.Queue(maxsize=256)
        self._agg_thread: Optional[threading.Thread] = None

        if self.config.aggregate:
            self.start_aggregator()

    # Aggregator worker thread
    def start_aggregator(self) -> None:
        self._agg_enabled = True
        self._agg_max_events = max(10, int(self.config.agg_max_events or 1000)) # 10 min, 1000 default
        if not self._agg_thread or not self._agg_thread.is_alive():
            self._agg_thread = threading.Thread(target=self._agg_exec, name="trace-agg", daemon=True)
            self._agg_thread.start()

    def stop_aggregator(self, *, flush: bool = True, timeout: float = 5.0) -> None:
        self._agg_enabled = False
        if flush and self._events:
            try:
                self._agg_q.put_nowait({
                    "req_id": (self._req_id or "run"),
                    "node_id": (self.config.node_id or "node"),
                    "events": list(self._events), })
            except queue.Full:
                logger.warning(f"Trace aggragator queue is full.")
            self._events.clear()
        if self._agg_thread and self._agg_thread.is_alive():
            self._agg_thread.join(timeout)
        self._agg_thread = None

    def _agg_exec(self) -> None:
        assert self.config.aggregate_url != ""
        client = httpx.Client(timeout=5.0)
        try:
            logger.debug(f"Aggregation worker thread {self._agg_enabled}, {self._agg_q.empty()}")
            while self._agg_enabled or not self._agg_q.empty():
                try:
                    batch = self._agg_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                logger.info(f"Sending trace buffer to API : {self.config.aggregate_url}")
                try:
                    res = client.post(self.config.aggregate_url, json=batch)
                    if res.status_code != 200:
                        logger.error(f"Aggregator POST failed {res.status_code}: {res.text}")
                except Exception as e:
                    logger.warning(f"Unable to POST trace aggregation data to {self.config.aggregate_url}: {e}")
                finally:
                    self._agg_q.task_done()
        finally:
            try:
                client.close()
            except Exception:
                logger.warining("Unable to close httpx client.")

    def update_api_addr(self, addr):
        self.config.aggregate_url = addr
        logger.debug(f"Updated API Address: {self.config.aggregate_url}")

    def start(self, *, reset: bool = True) -> None:
        self._active = bool(self.config.enabled)
        if not self._active:
            logger.info("Initialized tracer.")
            return
        if self.config.file: 
            d = os.path.dirname(self.config.file) or "."
            os.makedirs(d, exist_ok=True)
        if reset and os.path.exists(self.config.file):
            try:
                os.remove(self.config.file)
            except Exception:
                logger.warning(f"Unable to remove existing trace file {self.config.file}")
        if self.config.streaming:
            with self._lock:
                self._fh = open(self.config.file, "a", encoding="utf-8")
                logger.info(f"Streaming trace to {self.config.file}.")
        if self.config.aggregate and self.config.aggregate_url and self.config.node_id:
            self.start_aggregator()

    def stop(self, *, flush_events: bool = True) -> None:
        if flush_events:
            self.flush()
        self._active = False
        with self._lock:
            if self._fh:
                try:
                    self._fh.flush()
                    self._fh.close()
                except Exception:
                    logger.warning(f"Unable to flush to file {self.config.file}")
                self._fh = None

    # Flush file to disk
    def flush(self, *, clear: bool = False) -> None:
        if not self._active: return
        with self._lock:
            if not self.config.streaming and self._events:
                with open(self.config.file, "a", encoding="utf-8") as f:
                    for ev in self._events:
                        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                if clear:
                    self._events.clear()
                        
    # Quick dump to memory
    def snapshot(self, path: str) -> None:
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                for ev in self._events:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # emit a new frame
    def _emit(self, ev: Dict[str, Any]) -> None:
        if not self._active: return
        ev.setdefault("ts", time.perf_counter())
        if self._req_id is not None:
            ev.setdefault("req_id", self._req_id)
        if self.config.record_pid_tid:
            try:
                ev.setdefault("pid", os.getpid())
                ev.setdefault("tid", threading.get_ident())
            except Exception:
                logger.warning("Unable to get PID and TID for tracer frame.")

        with self._lock:
            if self.config.streaming and self._fh:
                self._fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                self._fh.flush()
            else:
                self._events.append(ev)

            if self._agg_enabled:
                if len(self._events) < self._agg_max_events: return
                logger.debug(f"Queuing tracer frame batch of {len(self._events)}")
                batch = { "run_id": (self._req_id or "NONE"),
                          "node_id": (self.config.node_id or "NODE"),
                          "events": list(self._events)}
                try:
                    self._agg_q.put_nowait(batch)
                except queue.Full:
                    logger.warning(f"Aggregator queue is full. Dropping {len(batch["events"])} frames.")
                self._events.clear()

    # Frames
    def frame(self, scope: str, name: str, attrs: Optional[Dict[str, Any]] = None):
        if not self._active:
            return _NoopFrame()
        return _Frame(self, f"{scope}.{name}", attrs) 

    # Same as normal frame but signals that this trace is a cannon event (required for runtime stats)
    def canonical(self, scope: str, name: str, attrs: Optional[Dict[str, Any]] = None):
      return self.frame(scope, name, attrs)

    # Mark an event outside of a frame
    def mark(self, name: str, attrs: Any = {}) -> None:
        self._emit({"type": "I", "name": name, "args": attrs})

    # Helpers
    @contextmanager
    def profile_block(self, outfile: Optional[str] = None, sort: str = "cumtime", limit: int = 40):
        pr = cProfile.Profile()
        pr.enable()
        try:
            yield pr
        finally:
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sort).print_stats(limit)
            out = s.getvalue()
            if outfile:
                d = os.path.dirname(outfile) or "."
                os.makedirs(d, exist_ok=True)
                with open(outfile, "w", encoding="utf-8") as f:
                    f.write(out)
            else:
                self._emit({"type": "PROFILE", "name": "cprofile", "attrs": {"sort": sort, "limit": limit, "report": out}})

    @contextmanager
    def callgraph(
        self,
        include_prefixes: Optional[Tuple[str, ...]] = None,
        budget_events: Optional[int] = None,
        include_c_calls: Optional[bool] = None,
        apply_to_new_threads: bool = False,
    ):
        """
        Interpreter-level tracing (sys.setprofile) for all Python calls/returns
        within the scope. Heavy overhead; best for deep debugging runs.
        """
        prefixes = include_prefixes if include_prefixes is not None else self.config.include_prefixes
        budget = (budget_events if budget_events is not None else self.config.budget) or 0
        inc_c = include_c_calls if include_c_calls is not None else self.config.include_c_calls

        emitted = 0
        stack: list[Tuple[str, float]] = []

        def prof(frame, event, arg):
            nonlocal emitted
            if budget and emitted >= budget:
                return
            if event in ("call", "return"):
                code = frame.f_code
                filename = code.co_filename or ""
                if prefixes and not any(filename.startswith(p) for p in prefixes):
                    return
                name = code.co_name
                key = f"{filename}:{code.co_firstlineno}:{name}"
                if event == "call":
                    stack.append((key, time.perf_counter()))
                    self._emit({"type": "B", "name": f"py.{name}", "attrs": {"file": filename, "line": code.co_firstlineno}})
                    emitted += 1
                else:
                    if stack and stack[-1][0] == key:
                        _, t0 = stack.pop()
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        self._emit({"type": "E", "name": f"py.{name}", "attrs": {"ms": round(dt_ms, 3)}})
                        emitted += 1
            elif inc_c and event in ("c_call", "c_return"):
                func = getattr(arg, "__name__", None)
                mod = getattr(arg, "__module__", None)
                if not func:
                    return
                if event == "c_call":
                    self._emit({"type": "B", "name": f"c.{mod}.{func}", "attrs": {}})
                    emitted += 1
                else:
                    self._emit({"type": "E", "name": f"c.{mod}.{func}", "attrs": {}})
                    emitted += 1

        prev = sys.getprofile()
        sys.setprofile(prof)
        prev_thread = None
        if apply_to_new_threads:
            prev_thread = threading.getprofile()
            threading.setprofile(prof)
        try:
            yield
        finally:
            sys.setprofile(prev)
            if apply_to_new_threads:
                threading.setprofile(prev_thread)
