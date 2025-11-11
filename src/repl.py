
import io
import os
import sys
import logging
import cmd
import time
import argparse
import subprocess
import contextlib
from dataclasses import dataclass
from typing import Optional, List, Any, Dict

import asyncio
import inspect
import threading
import concurrent.futures  

from dnet.ring.api import RingApiNode 
from dnet.ring.shard import RingShardNode 
from dnet.utils.network import NodeAddress
#from dnet.utils.logger import logger
from dnet.ring.api.api_logging import get_api_logger
from dnet.utils.model import ( 
  ModelMetadata, 
  get_model_metadata, 
  load_api_layer_weights,
  get_safetensor_details,
)

logger = get_api_logger()

from dnet.perf.trace import TraceConfig, Tracer
from dnet.perf.utils import TraceAggregator, StatsAggregator
#from dnet.perf.bench import 
from dnet.ring.common import TopologyInfo

from dnet.ring.api.models import (
  PrepareTopologyManualRequest,
  PrepareTopologyRequest,
  APILoadModelRequest,
  APILoadModelResponse,
)

# Handle restricted repos
from importlib import import_module
import huggingface_hub as hb
from huggingface_hub import snapshot_download, try_to_load_from_cache
try:
  hf_errors = import_module("huggingface_hub.errors")
except ModuleNotFoundError:
  hf_errors = import_module("huggingface_hub.utils")
GatedRepoError = getattr(hf_errors, "GatedRepoError", Exception)
HfHubHTTPError = getattr(hf_errors, "HfHubHTTPError", Exception)
LocalEntryNotFoundError = getattr(hf_errors, "LocalEntryNotFoundError", Exception)


def dprint(msg):
  sys.stdout.write(msg) 
  sys.stdout.flush()

@dataclass
class REPLState:
  model: str = None 
  model_info: ModelMetadata = None,
  num_local_nodes: int = 1
  running_port = 50501
  running_httpport = 8091 
  api_http_port: int = 8080 
  api_grpc_port: int = 50500 
  window_size = 2          # Number of layers per node per visit (also number resident in cache)
  topo: TopologyInfo = None


class REPL(cmd.Cmd):
  PS1 = "dnet > "
  WELCOME = "\nDNET Distributed Inference Engine, v0.1\nExperimental software. Type 'help' for usage hints.\n\n"

  def __init__(self, model=None, nodes=1):
    assert nodes >= 1 and nodes < 10, "Invalid number of local nodes. Must be 0 < num < 10."
    super().__init__()

    # State
    self.state = REPLState()
    self.state.model = model
    self.state.running_port += 2
    self.state.num_local_nodes = nodes

    # API Thread
    self._node: Optional[RingApiNode] = None
    self._api_thread: Optional[threading.Thread] = None 
    self._api_ready = threading.Event()
    self._api_running = threading.Event()
    self._api_searching = threading.Event() # Track mDNS searching 
    self._api_loop: Optional[asyncio.AbstractEventLoop] = None
    self._api_shutdown_e: Optional[asyncio.Event] = None
    self._api_exc: Optional[BaseException] = None

    # Tracing
    self._trace_cfg = TraceConfig(
      enabled=False,
      streaming=False,
      budget=3000,
      aggregate=True,
      agg_max_events=50,
    )
    self._tracing = threading.Event()
    self._tracer = None
    self._trace_file = f"trace-{time.strftime("%Y%m%d-%H%M%S")}" 
    self._trace_cursor = 0 # keep track of registered events in buffer 
    self._trace_agg = TraceAggregator()

    # Runtime stats (ingests data from tracer)
    self._stats_agg = StatsAggregator()
    self._stats = threading.Event()
    self._stats.set() # Trace runtime information by default


  def loop(self): # Main tty loop
    sys.stdout.write(self.WELCOME) 
    while True:
      dprint(self.PS1)
      cmd = sys.stdin.readline().strip().split(" ") 

      match cmd[0]:
        case "": continue
        case s if s in ["exit", "quit", "q"]:  self.handle_terminate_signal()
        case s if s in ["help", "h"]:          self.print_help()
        case s if s in ["api", "server"]:      self.do_api(cmd)
        case s if s in ["search", "s"]:        self.do_search(cmd)
        case s if s in ["nodes", "n"]:         self.print_mdns_nodes()
        case s if s in ["load", "l"]:          self.load_model(self.state.model)
        case s if s in ["trace", ".trace", "t"]:  self.do_trace(cmd)
        case s if s in ["perf", ".perf", "p"]:    self.do_perf(cmd)
        case s if s in ["topo", "topology", "t"]: self.do_topo(cmd)
        case s if s in ["model", "m"]:            self.do_model(cmd)

  def do_api(self, cmd: List[str]) -> None:
    if len(cmd) < 2:
      dprint("Invalid API command. Type 'help' for a list of valid commands.\n")
      return 
    match cmd[1]: # TODO Maybe allow kwargs here? > api start grpc_port=8080
      case s if s in ["start", "run"]:
        http_port, grpc_port = None, None
        if len(cmd) > 2:
          try:
            http_port = cmd[2];
            grpc_port = cmd[3]
          except:
            pass
        self.start_api( http_port or self.state.api_http_port, grpc_port or self.state.api_grpc_port)
        self.api_call("set_trace_ingest_callback", self.__trace_cb, timeout=2.0)

      case "stop":   self.stop_api()
      case "status": dprint("Running\n" if self._api_running else "Stopped.\n") 
      case "log":    dprint("Log print is not yet supported.\n")
      case _:        dprint("Invalid API command. Type 'help' for a list of valid commands.\n")
    return

  def do_search(self, cmd: List[str]) -> None:
    if len(cmd) != 2:
      dprint("mDNS search is " + ("ON\n\n" if self._api_searching else "OFF\n\n"))
      return 
    match cmd[1]: # NOTE on by default
      case "on":
        if self._api_searching:
          return
        if not self._api_ready and self._api_running:
          dprint("Starting API Server thread.\n")
          self.start_api()
        self.api_call("_start_discovery", timeout=10)
        self._api_searching.set()
        dprint("Starting mDNS search for worker nodes.\n")
      case "off": dprint("Stop discovery not yet implemented in the API node.\n")
      case _: dprint("Invalid topology command. Start searchign with 'search on'.\n")
    
  def do_topo(self, cmd: List[str]) -> None:
    if len(cmd) < 2:
      dprint("Invalid topology command. Type 'help' for a list of valid commands.\n")
      return 
    match cmd[1]:
      case "search": self.print_mdns_nodes()
      case s if s in ["auto", "build", "b"]: self.prepare_topo(self.state.model)
      case "add": dprint("Not implemented.\n")
      case s if s in ["remove", "rm"]: dprint("Not implemented.\n")

  # TODO: standardize ANSI escape codes for easy use
  def print_help(self):
    def _print_hf(cmd, desc, examples=[""]):
      pcmd = "    " + cmd.ljust(30, '.')
      sys.stdout.write(f"{pcmd} {desc}\n")
      for e in examples:
        pex = e.rjust(len(e)+35)+"\n" if e != "" else ""
        sys.stdout.write(f"{pex}")

    sys.stdout.write("\033[1m\nAvailable commands:\n\033[0m")
    dprint("\033[1m\n    Common:\n\033[0m")
    _print_hf("model list ", "List locally available models.")
    _print_hf("model [REPO]", "Set the target model. [REPO] must be a valid repository")
    _print_hf("nodes list ", "List mDNS discovered nodes.")
    _print_hf("log [LEVEL]", "Set the logging level.")
    dprint("\033[1m\n    Controlling the API Server:\n\033[0m")
    _print_hf("api start [HTTP] [GRPC]", "Start the API server in a separate thread. Use provided ports if given.")
    _print_hf("api stop ", "Stop the API server.")
    _print_hf("api status ", "Prints the status of the API server.")
    _print_hf("api log ", "Output live logs to current terminal.")
    dprint("\033[1m\n    Topology construction:\n\033[0m")
    _print_hf("search ", "Returns the current state of mDNS search.")
    _print_hf("search [on/off] ", "Toggle mDNS search across the local network.")
    _print_hf("nodes list ", "List all nodes in the current topology (including local ones).")
    _print_hf("nodes all ", "List all nodes (including local ones).")
    _print_hf("nodes ", "List mDNS discovered nodes.")
    _print_hf("topo [AUTO/SETUP]", "Toggle between automatic and manual topology creation.")
    _print_hf("topo add [NODE]", "Add [NODE] to the topology.")
    _print_hf("topo remove [NODE]", "Remove [NODE] from the topology.")
    sys.stdout.write("\033[1m\n    Scheduling:\n\033[0m")
    _print_hf("sched auto ", "Automatic search for best schedule given the active topology and the loaded model.")
    _print_hf("sched assign [INDEX] [NODE]", "Assign the layer range between [START] and [END] to [NODE].",
              ["Example: > sched assign 10 benny_234",
               "         > sched assign 0-12 benny_234"])
    sys.stdout.write("\033[1m\n    Benchmarking, Tracing and Profiling:\n\033[0m")
    _print_hf("trace [ON|OFF][PATH][SYSTEM] ", "Trace [SYSTEM] and output to file at [PATH].")
    _print_hf("trace status ", "See status of the trace, eg. number of frames captured")
    _print_hf("trace focus [SUBSYSTEM] ", "Focus the trace on [SUBSYSTEM]. Do 'trace focus' for a list of available subsystems.")
    _print_hf("trace stream [ON|OFF] ", "Stream the trace spans to current terminal.")
    _print_hf("trace set [BUDGET] ", "Set the maximum amount of recoded events.")
    _print_hf("perf ", "Prints the current state of runtime performance tracking.")
    _print_hf("perf stat [REQ_ID | WORKER_ID | MODEL] ", "Prints the runtime statistics of target system.")
    _print_hf("bench [REPO]", "Benchmark the system using the model from [REPO]")
    _print_hf("bench [KERNEL]", "Behcnmark the system using base kernel [KERNEL]")
    _print_hf("bench [NODE]", "Behcnmark the network latency between the current system and [NODE]")
    _print_hf("bench [NODE0] [NODE1]", "Behcnmark the network latency between [NODE0] and [NODE11")
    _print_hf("bench ", "Behcnmark the system using base library kernels")
    sys.stdout.write("\033[1m\n    System control:\n\033[0m")
    _print_hf("limit [RESOURCE] [VALUE]", "Set a higher limit for a system resource.",
              ["Examples  > limit memory 12000 (MB)",
               "          > limit CPU_CORE_COUNT 4",
               "          > limit GPU_SM 128"])
    sys.stdout.write("\n")
    sys.stdout.flush()
    
  def print_state(self):
    dprint("Network state:\n")
    dprint(f"{("Model".ljust(20)): >10}: {self.state.model}\n")
    dprint(f"{("Local workers".ljust(20)): >10}: {self.state.num_local_nodes}\n")


  # ===== Handle Model input and pull from server

  def prompt_model(self):
    while True:
      dprint("Target model > ")
      model = sys.stdin.readline().strip()
      try:
        path = self._handle_model_pull(model)
        return path
        #self.model_info = ModelMetadata()
      except Exception as e:
        dprint(f"Unable to load model {model}. Target needs to be a valid HF repository. Try again:{e}\n")

  def do_model(self, cmd):
    if len(cmd) < 2:
      if self.state.model is None: dprint("No target model.\n")
      else: dprint(f"Target model: {self.state.model}.\n")
      return

    match cmd[1]:
      case "list": # List locally available models
        lists = self._list_local_models()
        dprint("\nLocally available weights:\n")
        for x in lists[0]:
          dprint(f"   {x}\n")
        dprint("\nMetadata only:\n")
        for x in lists[1]:
          dprint(f"   {x}\n")
        dprint("\n")
      case _: # Treat unknown commands as model repos
        self._handle_model_pull(cmd[1])
    
  def _are_weights_local(self, repo_id, revision="main"):
    try:
      import re, os
      from pathlib import Path
      root = Path.home() / ".cache/huggingface/hub"
      if os.getenv("HF_HOME") is not None: root = Path(f"{os.getenv("HF_HOME")}/huggingface/hub") 
      if os.getenv("HF_HUB_CACHE") is not None: root = Path(f"{os.getenv("HF_HUB_CACHE")}/huggingface/hub")
      files = os.scandir(root / f"models--{repo_id}/snapshots")
      commit_hash = next(files).name 
      files = os.scandir(root / f"models--{repo_id}/snapshots/{commit_hash}")
      files = [x.name for x in files]
    except Exception as e:
      dprint(f"Failed to search files in local model cache: {e}\n")
      return False
    PATTERNS = [
        re.compile(r".*?model.*\.safetensors$"),
        re.compile(r"^(model|pytorch_model)([._-]\d+([-_]of[-_]\d+)?)?[._-]?\.safetensors$"),
        re.compile(r"pytorch_model[-_]\d{3,}[.]bin$"),
        re.compile(r"model[.]safetensors$"),          
        re.compile(r"pytorch_model[.]bin$"),          
        re.compile(r"tf_model[.]h5$"),                
        re.compile(r"flax_model[.]msgpack$"),         
    ]
    for pat in PATTERNS:
      if any(pat.search(f) for f in files):
        return True
    return False

  def _list_local_models(self):
    import os
    from pathlib import Path
    root = Path.home() / ".cache/huggingface/hub"
    if os.getenv("HF_HOME") is not None: root = Path(f"{os.getenv("HF_HOME")}/huggingface/hub") 
    if os.getenv("HF_HUB_CACHE") is not None: root = Path(f"{os.getenv("HF_HUB_CACHE")}/huggingface/hub")
    models = [x.name.replace("models--", "") for x in os.scandir(root) if x.name.startswith("models--")]
    weights_local = sorted([x for x in models if self._are_weights_local(x)], key=len)
    config_local = sorted([x for x in models if x not in weights_local], key=len)
    return [weights_local, config_local]

  # Require a HF access token for restricted repositories
  # Ask user for HF access token until they have a valid one
  def _handle_model_pull(self, repo_path):
    local = self._are_weights_local(repo_path)
    try:
      if not local:
        dprint(f"Weights for {repo_path} not found in local cache. Downloading.\n")
        path = snapshot_download(repo_path)
        dprint("Download complete\n")
      else:
        dprint("Target model found in local registry.\n")

      self.state.model = repo_path
      return repo_path 

    except GatedRepoError as e:
      tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
      while True:
        dprint("\nRestricted model. Ener the HuggingFace access token > ")
        tok = sys.stdin.readline().strip()
        try:
          ret = snapshot_download(repo_id=repo_path, token=tok) 
          self.state.model = path
          return ret
        except GatedRepoError as e:
          print(e)
          continue
        except Exception as e:
          raise RuntimeError(f"Unknown error during HF snapshot_download")

    except HfHubHTTPError as e:
      dprint(f"Repository {repo_path} not found in Hugging Face registry: {e}")
      return None 

    except Exception as e:
      raise RuntimeError(f"Unable to pull model {repo_path} locally")

  def _parse_model_metadata(self, model_path):
    if isinstance(model_path, tuple):
      model_path = model_path[0] if model_path else None
    if model_path is None:
      raise ValueError("Could not resolve model path {model_path}")
    path = Path(model_path) if not isinstance(model_path, Path) else model_path

    with open(path / "config.json", "r") as f:
      config = json.load(f)

    weight_files = glob.glob(oath / "*.safetensors")
    weight_info = Dict[int, Dict[str, Any]] = defaultdict(dict)
    embed_tokens, lm_head, norm = {}, {}, {}
    for weight in weight_files:
        details = get_safetensor_details(weight)
        for key, val in details.items():
            if m := EMBED_TOKENS_RE.match(key):
                embed_tokens[m.group(1)] = val
            elif m := LM_HEAD_RE.match(key):
                lm_head[m.group(1)] = val
            elif m := NORM_RE.match(key):
                norm[m.group(1)] = val
            elif m := LAYERS_RE.match(key):
                layer_idx, suffix = m.groups()
                weight_info[int(layer_idx)][suffix] = val
            else:
                raise RuntimeError(f"Unexpected key {key}")
    num_layers = max(weight_info.keys()) + 1
    if not (set(weight_info.keys()) == set(range(num_layers))):
        raise RuntimeError("Inconsistent weights")
    return ModelMetadata(path, weight_info, embed_tokens, lm_head, norm, config)
    
  # ===== Handle termination signals

  def handle_terminate_signal(self):
    # Handle worker/api shutdown
    if self._api_running:
      self.stop_api()
    else:
      dprint("No workers to shut down. Terminating.\n")
    sys.exit()

  # ===== Handle Shard worker servers 

  # TODO: Redirect output logs to different files
  def handle_start_worker(self):
    bin = os.path.join(".venv", "bin", "piped_mlx_ring_shard")
    cmd = ["uv", "run", bin]
    cmd.append(f" -m {self.state.model}")
    cmd.append(f" -p {self.state.running_port}")
    cmd.append(f" -httpport {self.state.running_httpport}")
    cmd.append(f" -l [{0}]")
    cmd.append(f" --prefetch-window {2}")
    proc = subprocess.Popen(cmd)

    self.state.running_port += 1 # increment the running port
    self.state.running_httpport += 1 

  # ===== Handle API server

  async def _api_main(self) -> None: # main thread loop
    #logging.disable(logging.CRITICAL)
    self._api_loop = asyncio.get_running_loop()
    self._api_shutdown_e = asyncio.Event()
    self._node = RingApiNode(
      http_port=self.state.api_http_port, 
      grpc_port=self.state.api_grpc_port
    )

    try:
      await self._node.start(shutdown_trigger=self._api_shutdown_e.wait)
      self._api_searching.set()
      self._api_running.set()
      self._api_ready.set()
      await self._api_shutdown_e.wait()
    except Exception as e:
      self._api_exc = e
      self._api_running.set()
      self._api_ready.set()
    finally:
      try:
        await self._node.shutdown()
      except Exception:
        pass
      self._api_running.clear()

  def _api_running_loop(self): 
    try:
      asyncio.run(self._api_main())
    except BaseException as e:
      self._api_exc = e
      self._api_ready.set()
      self._api_running.clear()

  def start_api(self, http_port: int=8080, grpc_port: int=50500, timeout=10):
    if self._api_thread and self._api_thread.is_alive(): return
    self._api_exc = None
    self._api_ready.clear()
    self._api_running.clear()
    self._api_thread = threading.Thread(target=self._api_running_loop, name="api_server", daemon=True)
    self._api_thread.start()
    if not self._api_ready.wait(timeout):
      raise RuntimeError("API Server Timeout.")
    if self._api_exc is not None:
      raise RuntimeError(f"API Server failed to start: {self._api_exc}")
    # Register REPL aggregator callback on the API node
    try:
      self.api_call("set_trace_ingest_callback", self._trace_agg.enqueue, timeout=5)
    except Exception:
      pass

    # Silence API server logs on the REPL console: drop records emitted from the API thread
    try:
      class _DropApiOnConsole(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
          # Only drop records coming from the API thread so other threads keep logging
          tname = getattr(record, "threadName", "") or ""
          return tname != "api_server"

      root = logging.getLogger()
      for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (sys.stdout, sys.stderr):
          if not any(isinstance(f, _DropApiOnConsole) for f in getattr(h, "filters", [])):
            h.addFilter(_DropApiOnConsole())

      # Also quiet Hypercorn logs explicitly (HTTP server used by API)
      logging.getLogger("hypercorn").setLevel(logging.CRITICAL)
      logging.getLogger("hypercorn.error").setLevel(logging.CRITICAL)
      logging.getLogger("hypercorn.access").setLevel(logging.CRITICAL)
    except Exception:
      pass

  def stop_api(self, timeout: float = 5.0) -> None:
    if not self._api_thread: return
    if self._api_loop and self._api_shutdown_e:
      self._api_loop.call_soon_threadsafe(self._api_shutdown_e.set)
    if self._api_loop and self._node:
      f = asyncio.run_coroutine_threadsafe(self._node.shutdown(), self._api_loop)
      try:
        f.result(timeout=timeout)
      except Exception:
        pass
    self._api_thread.join(timeout=timeout)
    self._api_thread = None
    self._api_running.clear()
    self._api_ready.clear()

  # Call an API function from the REPL thread
  def api_call( self, method: str, *args: Any, timeout: float=30.0, **kwargs: Any) -> Any:
    if not self._api_loop or not self._node:
      raise RuntimeError("API Thread not set up correctly.")

    target = getattr(self._node, method, None)
    if target is None:
      raise AttributeError(f"RingApiNode has no method {method}")

    # method is async 
    if inspect.iscoroutinefunction(target):
      coroutine = target(*args, **kwargs)
      f = asyncio.run_coroutine_threadsafe(coroutine, self._api_loop) 
      return f.result(timeout)

    f = concurrent.futures.Future() # method is sync

    def runner():
      try:
        ret = target(*args, **kwargs)
        if inspect.isawaitable(ret):

          async def _await_then_set():
            try:
              val = await res
              f.set_result(val)
            except BaseException as e:
              f.set_exception(e)
            
          asyncio.create_task(_await_then_set())
        else:
          f.set_result(ret)
      except BaseException as e:
        f.set_exception(e)
    try:
      self._api_loop.call_soon_threadsafe(runner)
      return f.result(timeout)
    except Exception as e:
      raise 

  # ------- Trace aggregation helpers 

  def do_trace(self, cmd):
    if len(cmd) < 2:
      dprint(f"Tracing is currently {"ON" if self._trace_cfg.enabled else "OFF"}\n")
      return
    
    match cmd[1]:
      case s if s in ["on", "ON"]:
        self._tracing.set()
        self._trace_cfg.enabled = True
        dprint("Tracing is now ON\n")

      case s if s in ["off", "OFF"]:
        self._tracing.clear()
        self._trace_cfg.enabled = False 
        dprint("Tracing is now OFF\n")

      case s if s in ["status"]:
        dprint(f"Captured {len(self._trace_agg._req)} frames.\n")

      case s if s == "focus":
        dprint("Subsystems not yet implemented.\n")

      case s if s == "stream":
        if len(cmd) == 2:
          dprint(f"Trace is {"streaming to file: "+str(self._trace_cfg.file) if self._trace_cfg.streaming else "not streaming."}\n")
        elif cmd[2] == "on":
          self._trace_cfg.streaming = True
          dprint(f"Streaming trace frames to {self._trace_cfg.file}\n")
        elif cmd[2] == "off":
          self._trace_cfg.streaming = False 
          dprint("Trace streaming is OFF.\n")

      case s if s == "set":
        if len(cmd) == 2:
          dprint("Use: trace set [BUDGET], eg. 2000\n")
        else:
          dprint("Not implemented yet\n")

      case s if s == "annotate":
        if len(self._trace_agg._req) < 1:
          dprint("No trace frames captured. Is tracing enabled?\n")
          return
        last = list(self._trace_agg._req.keys())[-1]
        self.print_trace_annotate(last)

      case _:
        dprint("Unknown trace command. Type 'help' for a list of available commands.\n")
          
    if self._api_running.is_set() and self._api_ready.is_set():
      self.api_call("_forward_trace_config", self._trace_cfg) # Send trace config to all shards

  # Performance trackers 
  def do_perf(self, cmd):
    if len(cmd) < 2 or cmd[1] == "stat":
      dprint("Runtime performance metrics are ON by default.\n") 
      dprint("Turn tracking off with 'perf off'. Do 'perf stat' for statistics on previous requests or 'help' for more commands.\n\n")
      return

    match cmd[1]:
      case s if s in "stats":
        self._stats_agg.stats()
        pass
      case _:
        pass

  # Trace callback registered with API Thread
  # This forwards the tracer frames back to the REPL for printing
  def __trace_cb(self, data):
    try:
      if self._tracing.is_set():
        self._trace_agg.enqueue(data)
      if self._stats.is_set():
        self._stats_agg.add(data)
    except Exception as e:
      print(f"Unable to ingest trace buffer into REPL: {e}")

  def __print_tr(self, row):
    sym = "    " + symbol.ljust(40, ' ')
    pms = f"{ms:.10}".ljust(10, ' ') 
    cns = f"{counts}".ljust(4, ' ')
    sys.stdout.write(f"{sym} {pms} {cns}\n")

  def print_trace_annotate(
    self,
    run_id: str = "run",
    mapping: Optional[Dict[str, str]] = None,
    repeats: int = 0,
  ) -> List[Dict[str, Any]]:

    try:
      rows = self._trace_agg.annotate(run_id)
      logger.debug(f"rows")
      headers = ["name", "total","max","mean","p50","p90","p99","samples"]
      limits = {"name": 50,}
      w = {h: max(len(h), min(limits.get(h, 8), max(len(str(r[h])) for r in rows))) for h in headers}
      w["name"] = max(w["name"], 35)

      line = "  ".join(h.ljust(w[h]) for h in headers); sys.stdout.write("\n")
      sys.stdout.write(line + "\n")
      sys.stdout.write("  ".join("."*w[h] for h in headers)); sys.stdout.write("\n")
      for r in rows:
        name = str(r["name"])
        if len(name) > w["name"]: name = name[:w["name"]-1] + "..."
        vals = {
          "name": r["name"],
          "total": r["total"],
          "max": r["max"],
          "mean": r["mean"],
          "p50": r["p50"],
          "p90": r["p90"],
          "p99": r["p99"],
          "samples": r["samples"], 
        }
        sys.stdout.write("  " + str(vals[headers[0]]).ljust(w[headers[0]]))
        sys.stdout.write("  ".join(f"{vals[h]:8.2f}".rjust(w[h]) for h in headers[1:]))
        sys.stdout.write("\n")
      sys.stdout.write("\n\n")
      sys.stdout.flush()
    except Exception as e:
      logger.error(f"{e}")

  # ------- Trace aggregation helpers 

  def do_trace(self, cmd):
    if len(cmd) < 2:
      dprint(f"Tracing is currently {"ON" if self._trace_cfg.enabled else "OFF"}\n")
      return
    
    match cmd[1]:
      case s if s in ["on", "ON"]:
        self._tracing.set()
        self._trace_cfg.enabled = True
        dprint("Tracing is now ON\n")

      case s if s in ["off", "OFF"]:
        self._tracing.clear()
        self._trace_cfg.enabled = False 
        dprint("Tracing is now OFF\n")

      case s if s in ["status"]:
        dprint(f"Captured {len(self._trace_agg._req)} frames.\n")

      case s if s == "focus":
        dprint("Subsystems not yet implemented.\n")

      case s if s == "stream":
        if len(cmd) == 2:
          dprint(f"Trace is {"streaming to file: "+str(self._trace_cfg.file) if self._trace_cfg.streaming else "not streaming."}\n")
        elif cmd[2] == "on":
          self._trace_cfg.streaming = True
          dprint(f"Streaming trace frames to {self._trace_cfg.file}\n")
        elif cmd[2] == "off":
          self._trace_cfg.streaming = False 
          dprint("Trace streaming is OFF.\n")

      case s if s == "set":
        if len(cmd) == 2:
          dprint("Use: trace set [BUDGET], eg. 2000\n")
        else:
          dprint("Not implemented yet\n")

      case s if s == "annotate":
        if len(self._trace_agg._req) < 1:
          dprint("No trace frames captured. Is tracing enabled?\n")
          return
        last = list(self._trace_agg._req.keys())[-1]
        self.print_trace_annotate(last)

      case _:
        dprint("Unknown trace command. Type 'help' for a list of available commands.\n")
          
    if self._api_running.is_set() and self._api_ready.is_set():
      self.api_call("_forward_trace_config", self._trace_cfg) # Send trace config to all shards

  # Performance trackers 
  def do_perf(self, cmd):
    if len(cmd) < 2 or cmd[1] == "stat":
      dprint("Runtime performance metrics are ON by default.\n") 
      dprint("Turn tracking off with 'perf off'. Do 'perf stat' for statistics on previous requests or 'help' for more commands.\n\n")
      return

    match cmd[1]:
      case s if s in "stats":
        self._stats_agg.stats()
        pass
      case _:
        pass

  # Trace callback registered with API Thread
  # This forwards the tracer frames back to the REPL for printing
  def __trace_cb(self, data):
    try:
      if self._tracing.is_set():
        self._trace_agg.enqueue(data)
      if self._stats.is_set():
        self._stats_agg.add(data)
    except Exception as e:
      print(f"Unable to ingest trace buffer into REPL: {e}")

  def __print_tr(self, symbol, ms, counts):
    sym = "    " + symbol.ljust(40, ' ')
    pms = f"{ms:.10}".ljust(10, ' ') 
    cns = f"{counts}".ljust(4, ' ')
    sys.stdout.write(f"{sym} {pms} {cns}\n")

  def print_trace_annotate(
    self,
    run_id: str = "run",
    mapping: Optional[Dict[str, str]] = None,
    repeats: int = 0,
  ) -> List[Dict[str, Any]]:
    names = " "*17 + "symbol" + " "*21 + "ms" + " "*4 + "counts"
    dots = "   " + "."*41 + " " + "."*10 + " " + "."*4
    dprint(f"{names}\n{dots}\n\n")
    sums = self._trace_agg._req[run_id].sums_by_name
    cnts = self._trace_agg._req[run_id].counts_by_name
    for n, d in sums.items():
      self.__print_tr(n, d, cnts[n])

  def get_trace_roots(self, run_id: str, req_id: str) -> List[Dict[str, Any]]:
    return self._trace_agg.roots(run_id, req_id)

  def _print_nodes_table(self, rows: List[Any]) -> None:
    headers = ["name", "role", "addr", "http", "grpc", "status", "head"]
    limits = {"name": 36, "addr": 15}
    w = {h: max(len(h), min(limits.get(h, 8), max(len(str(r[h])) for r in rows))) for h in headers}
    line = "  ".join(h.ljust(w[h]) for h in headers)
    sys.stdout.write("\n")
    sys.stdout.write(line + "\n")
    sys.stdout.write("  ".join("-"*w[h] for h in headers))
    sys.stdout.write("\n")
    for r in rows:
      name = str(r["name"])
      addr = str(r["addr"])
      if len(name) > w["name"]: name = name[:w["name"]-1] + "..."
      if len(addr) > w["addr"]: addr = addr[:w["addr"]-1] + "..."
      vals = {
        "name": name,
        "role": r["role"],
        "addr": addr,
        "http": r["http"],
        "grpc": r["grpc"],
        "status": "yes" if r["status"] else "no",
        "head": "head" if r["head"] else "no",
      }
      sys.stdout.write("  ".join(str(vals[h]).ljust(w[h]) for h in headers))
      sys.stdout.write("\n")
    sys.stdout.write("\n\n")
    sys.stdout.flush()
      

  # Print a table of discovered nodes 
  def print_mdns_nodes(self) -> None:
    try:
      shards = self.api_call("_get_shards_from_discovery", timeout=10)
      if not shards:
        dprint("No worker nodes discovered. Is the API searching?\n")
        return

      rows = []
      for name, props in shards.items():
        addr = getattr(props, "local_ip", getattr(props, "host", ""))
        http = getattr(props, "server_port", 0)
        grpc = getattr(props, "shard_port", 0)
        busy = bool(getattr(props, "is_busy", False))
        head = bool("127.0.0.1" and "127.0.0.1" == addr) # TODO: FIX
        rows.append({
          "name": name,
          "role": "worker",
          "addr": addr,
          "http": http,
          "grpc": grpc,
          "status": busy,
          "head": head,
        })
      self._print_nodes_table(rows)
    except Exception as e:
      dprint(f"Could not list nodes: {e}\n")

  def print_topo(self, topo):
    line = "="*20+" Topology " + "="*20
    sys.stdout.write(f"{line}\nModel: {topo.model}\nLayers: {topo.num_layers}\n")
    sys.stdout.write(f"Devices: {topo.devices}\n\n")
    # TODO: Better print here

  def prepare_topo(self, model):
    req = PrepareTopologyRequest(model=model)
    try:
      topo = self.api_call("_handle_prepare_topology", req, timeout=120)
    except Exception as e:
      dprint(f"Unable to create topology: {e}\n\n")
      return False
    self.state.topo = topo
    self.print_topo(topo)
    return True

  def load_model(self, model):
    req = APILoadModelRequest(model=model)
    try:
      res = self.api_call("_handle_load_model", req, timeout=30)
      return True 
    except Exception as e:
      dprint(f"Failed to load model: {e}\n\n")
      return False
    
  # ===== Handle chat
 
  def do_chat(self, cmd):
    model = "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"
    if len(cmd) < 2:
      if not self.state.model or self.state.model == "":
        self.prompt_model()
      if not self.state.topology:
        if not self._prepare_topo(self.state.model):
          raise RuntimeError("Unable to create topology.")
      if not self.load_model(self.state.model):
        raise RuntimeError("Unable to load model.")

      while True:
        prompt = input("\n> ")
        prompt = self.format_prompt(prompt)
        messages = prompt
        req = ChatRequest(
          messages=messages,
          max_tokens=100,
          temperature=0.7,
          stream=True,
        )

        self.api_call("_handle_completion", req)
        
      # Start default chat with selected model
      pass
    pass

  # ===== Handle shutdown 

  def handle_shutdown(self):
    os.system("pkill -9 -f piped_mlx")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", type=str, help="HF Repository of target model")
  parser.add_argument("--local-nodes", "-n", type=int, help="Number of local worker nodes")
  args = parser.parse_args()

  #workers = args.workers
  #model = args.model

  repl = REPL()
  repl.loop()
