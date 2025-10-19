"""
Object-oriented tracing utilities for dnet.

This module provides a Tracer class configured explicitly from the REPL (or code),
without relying on environment variables or module-level globals. It supports:

- Boundary frames via tracer.frame(scope, name, attrs)
- Deep sys.setprofile callgraph via tracer.callgraph(...)
- Aggregated call stats via tracer.profile_block(...)

All events are written as JSON Lines to a file (TraceConfig.file), suitable
for simple REPL visualization and easy sharing.
"""

from __future__ import annotations

import os
import sys
import time
import json
import threading
import contextvars
import cProfile
import pstats
import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from contextlib import contextmanager

from dnet.utils.logger import logger

@dataclass
class TraceConfig:
    file: str = "logs/dnet-trace.jsonl"
    streaming: bool = True
    include_prefixes: Tuple[str, ...] = ("src/dnet/",)
    include_c_calls: bool = False
    budget: int = 0  # 0 means unlimited
    enabled: bool = True
    record_pid_tid: bool = True


class Tracer:
    def __init__(self, cfg: TraceConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._fh: Optional[io.TextIOBase] = None
        self._events: List[Dict[str, Any]] = []
        self._req_id: str = None
        self._active = False

    def start(self, *, reset: bool = True) -> None:
        self._active = bool(self.cfg.enabled)
        if not self._active:
            logger.info("Initialized tracer.")
            return
        if self.cfg.file: 
          d = os.path.dirname(self.cfg.file) or "."
          os.makedirs(d, exist_ok=True)
        if reset and os.path.exists(self.cfg.file):
            try:
                os.remove(self.cfg.file)
            except Exception:
                logger.warning(f"Unable to remove existing trace file {self.cfg.file}")
        if self.cfg.streaming:
            with self._lock:
                self._fh = open(self.cfg.file, "a", encoding="utf-8")
                logger.info(f"Streaming trace to {self.cfg.file}.")

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
                    logger.warning(f"Unable to flush to file {self.cfg.file}")
                self._fh = None

    def set_request_id(self, rid: Optional[str]) -> None:
        self._req_id = rid

    def get_request_id(self) -> Optional[str]:
        return self._req_id

    # Flush file to disk
    def flush(self, *, clear: bool = False) -> None:
        if not self._active:
            return
        with self._lock:
            if not self.cfg.streaming and self._events:
                with open(self.cfg.file, "a", encoding="utf-8") as f:
                    for ev in self._events:
                        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                if clear:
                    self._events.clear()

    def snapshot(self, path: str) -> None:
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                for ev in self._events:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # Emit a new frame event
    def _emit(self, ev: Dict[str, Any]) -> None:
        if not self._active:
            return
        ev.setdefault("ts_us", time.time_ns() // 1000)
        if self._req_id is not None:
            ev.setdefault("req_id", self._req_id)
        if self.cfg.record_pid_tid:
            try:
                ev.setdefault("pid", os.getpid())
                ev.setdefault("tid", threading.get_ident())
            except Exception:
                logger.warning("Unable to get PID and TID for tracer frame.")
        with self._lock:
            if self.cfg.streaming and self._fh:
                self._fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                self._fh.flush()
            else:
                self._events.append(ev)

    # Frames
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
            self._t0 = time.perf_counter()
            self.t._emit({"type": "B", "name": self.name, "args": dict(self.attrs)})
            return self
        def __exit__(self, ex_type, ex, tb):
            dt_ms = (time.perf_counter() - self._t0) * 1000.0
            self.t._emit({"type": "E", "name": self.name, "args": {"ms": round(dt_ms, 3), "exc": bool(ex)}})
            return False
        def event(self, name: str, **attrs):
            self.t._emit({"type": "I", "name": f"{self.name}.{name}", "args": attrs})
        def set(self, key: str, val: Any):
            self.attrs[key] = val

    def frame(self, scope: str, name: str, attrs: Optional[Dict[str, Any]] = None):
        if not self._active:
            return Tracer._NoopFrame()
        return Tracer._Frame(self, f"{scope}.{name}", attrs) 

    def mark(self, name: str, **attrs: Any) -> None:
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
                self._emit({"type": "PROFILE", "name": "cprofile", "args": {"sort": sort, "limit": limit, "report": out}})

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
        prefixes = include_prefixes if include_prefixes is not None else self.cfg.include_prefixes
        budget = (budget_events if budget_events is not None else self.cfg.budget) or 0
        inc_c = include_c_calls if include_c_calls is not None else self.cfg.include_c_calls

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
                    self._emit({"type": "B", "name": f"py.{name}", "args": {"file": filename, "line": code.co_firstlineno}})
                    emitted += 1
                else:
                    if stack and stack[-1][0] == key:
                        _, t0 = stack.pop()
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        self._emit({"type": "E", "name": f"py.{name}", "args": {"ms": round(dt_ms, 3)}})
                        emitted += 1
            elif inc_c and event in ("c_call", "c_return"):
                func = getattr(arg, "__name__", None)
                mod = getattr(arg, "__module__", None)
                if not func:
                    return
                if event == "c_call":
                    self._emit({"type": "B", "name": f"c.{mod}.{func}", "args": {}})
                    emitted += 1
                else:
                    self._emit({"type": "E", "name": f"c.{mod}.{func}", "args": {}})
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

