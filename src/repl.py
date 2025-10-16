
import os
import sys
import cmd
import argparse
import subprocess
from dataclasses import dataclass

from src.ring.api import run as run_api_node
from src.ring.shard import run as run_shard_node
from src.util import (
  ModelMetadata,
  NodeAddress,
  logger,
  get_model_metadata,
  load_api_layer_weights,
  get_safetensor_details,
  create_generate_step_for_ring_with_grpc,
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

from src.ring.api_node import RingApiNode

def dprint(msg):
  sys.stdout.write(msg) 
  sys.stdout.flush()


@dataclass
class REPLState:
  model: str = "NULL"
  model_info: ModelMetadata = None,
  num_local_nodes: int = 1
  running_port = 50501
  running_httpport = 8091 
  api_addr_host: str = "10.0.0.2" # TODO: Don't hardcode
  api_addr_port: int = 0 
  grpc_listen_port:int = 0 
  window_size = 2          # Number of layers per node per visit (also number resident in cache)

class REPL(cmd.Cmd):

  PS1 = "dnet > "
  WELCOME = "\nDNET Distributed Inference Engine, v0.1\nExperimental software. Enter '.help' for usage hints.\n\n"
  def __init__(self, model="NULL", nodes=1):
    super().__init__()
    self.state = REPLState()
    self.state.model = model

    self.state.api_addr_port = self.state.running_port
    self.state.grpc_listening_port = self.state.running_port + 1 
    self.state.running_port += 2
    self.discovery = None

    # TODO: Maybe have a 'start search' 'stop search' cmds to manage discovery

    self.api = None
    #self.config_api_node()
    #self.start_api_discovery()

    assert nodes >= 1 and nodes < 10, "Invalid number of local nodes. Must be 0 < num < 10."
    self.state.num_local_nodes = nodes

  def loop(self):
    self.greeting()
    while True:

      #if self.state.model == "NULL":
      #  self.prompt_model()
      #  continue

      dprint(self.PS1)
      cmd = sys.stdin.readline().strip() 

      if cmd == "":
        self.print_state()
      if cmd in [".exit", "exit", "quit", "q"]:
        self.handle_terminate_signal()
      if cmd in [".help", "help", "h"]:
        self.print_help()
      if cmd.startswith((".model", "model", "m")):
        cmd.split(" ")
        path = self._handle_model_pull(cmd[1])
        if path:
          self.state.model = path

  def greeting(self):
    sys.stdout.write(self.WELCOME) 

  def print_help(self):
    def _print_hf(cmd, desc, examples=[""]):
      pcmd = "    " + cmd.ljust(30, '.')
      dprint(f"{pcmd} {desc}\n")
      for e in examples:
        pex = e.rjust(len(e)+35)+"\n" if e != "" else ""
        dprint(f"{pex}")

    dprint("Command Options:\n")
    _print_hf("nodes [VALUE]", "Set the number of local worker nodes")
    _print_hf("model [REPO]", "Set the target model. [REPO] must be a valid repository",
              ["Examples  > model meta-llama/Meta-Llama-3-8B"])
    _print_hf("limit [RESOURCE] [VALUE]", "Set a higher limit for a system resource.",
              ["Examples  > limit memory 12000 (MB)",
               "          > limit CPU_CORE_COUNT 4",
               "          > limit GPU_SM 128"])
    _print_hf("log [LEVEL]", "Set the logging level.")
    dprint("\n    Building a topology:\n")
    _print_hf("search [ON/OFF]", "Toggle mDNS worker node search across the local network.")
    _print_hf("topo [AUTO/SETUP]", "Toggle between automatic and manual topology creation.")
    _print_hf("topo add [NODE]", "Add [NODE] to the topology.")
    _print_hf("topo remove [NODE]", "Add [NODE] to the topology.")
    dprint("\n    Building a schedule:\n")
    _print_hf("sched create", "Automatic search for best schedule given the active topology and the loaded model.")
    _print_hf("sched assign [LAYER] [NODE]", "Assign the layer with index [LAYER] to [NODE].",
              ["Example   > sched assign 10 benny_234"])
    _print_hf("schedule assign [START-END] [NODE]", "Assign the layer range between [START] and [END] to [NODE].",
              ["Example   > sched assign 0-12 benny_234"])
    dprint("\n    Benchmarking and profiling:\n")
    _print_hf("profile [REPO]", "Estimate the total FLOPS of the model from [REPO]")
    _print_hf("bench [REPO]", "Benchmark the system using the model from [REPO]")
    _print_hf("bench [KERNEL]", "Behcnmark the system using base kernel [KERNEL]")
    _print_hf("bench [NODE]", "Behcnmark the network latency between the current system and [NODE]")
    _print_hf("bench [NODE0] [NODE1]", "Behcnmark the network latency between [NODE0] and [NODE11")
    _print_hf("bench ", "Behcnmark the system using base library kernels")
    dprint("\n")
    
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

  # Read HF access token
  def _resolve_hf_token(self):
    dprint("Ener the HuggingFace access token > ")
    tok = sys.stdin.readline().strip()
    return tok

  # Require a HF access token for restricted repositories
  # Ask user for HF access token until they have a valid one
  def _handle_model_pull(self, repo_path):
    try:
      path = try_to_load_from_cache(repo_path)
      if path is None:
        dprint(f"Model {repo_path} not found in local cache\n")
        path = get_model_path(repo_path)
      self.state.model = repo_path
      return path 
    except hb.errors.HTTPError:
      dprint(f"Repository {repo_path} not found in Hugging Face registry.")
      return Null 
    except GatedRepoError as e:
      dprint("Restricted model.\n")
      tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
      while True:
        tok = self._resolve_hf_token()
        print(tok)
        try:
          ret = snapshot_download(repo_id=repo_path, token=tok) 
          return ret
        except GatedRepoError as e:
          print(e)
          continue
        except Exception as e:
          raise RuntimeError(f"Unknown error during HF snapshot_download")
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

  def handle_device_discovery(self):
    from socket import gethostname
    from secrets import token_hex

    hostname = gethostname()
    instance = f"api-{token_hex(4)}-{hostname}"
    lib = DnetP2P("lib/dnet-p2p/lib")  

    """
    self.discovery = lib.create_instance(
      instance, hostname,
      self.state.p2p_addr.host, self.state.p2p_addr_port,
      self.state.grpc_listen_port, is_manager=True 
    ) 
    self.discovery.start()
    """

  def config_api_node(self):
    api_address = NodeAddress(self.state.api_addr_host, self.state.api_addr_port)
    self.api = RingApiNode(api_address, shard_address.format(), model_metadata)

  def start_api_discovery(self):
    if self.api:
      self.api._start_discovery()

  # Calls dsolver and optimizes topology
  async def build_topology(self):
    if self.api:
      topo = await self.api.topology()
      return topo

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
