"""API-facing fakes: HTTP client wrappers, tokenizers, managers, inference."""

from __future__ import annotations

from typing import Any, Dict, Callable
import httpx
import json as jsonlib
import types


class FakeClient:
    """Async httpx-like client using in-memory handler maps."""

    def __init__(
        self,
        get_map: Dict[str, Callable[[], Any]],
        post_map: Dict[str, Callable[[Any], Any]],
    ):
        self.get_map = get_map
        self.post_map = post_map

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, timeout: float = 5.0):
        fn = self.get_map.get(url)
        if isinstance(fn, Exception):
            raise fn
        resp = fn()
        if isinstance(resp, httpx.Response):
            return resp
        if isinstance(resp, FakeResponse):
            return httpx.Response(
                status_code=resp.status_code,
                content=jsonlib.dumps(resp._payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                request=httpx.Request("GET", url),
            )
        return httpx.Response(status_code=200, request=httpx.Request("GET", url))

    async def post(self, url: str, json=None, timeout: float = 1000.0):
        fn = self.post_map.get(url)
        if isinstance(fn, Exception):
            raise fn
        resp = fn(json)
        if isinstance(resp, httpx.Response):
            return resp
        if isinstance(resp, FakeResponse):
            return httpx.Response(
                status_code=resp.status_code,
                content=jsonlib.dumps(resp._payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                request=httpx.Request("POST", url),
            )
        return httpx.Response(status_code=200, request=httpx.Request("POST", url))


class FakeResponse:
    """Small response placeholder with .json() method."""

    def __init__(self, code: int, payload: Dict[str, Any] | None = None):
        self.status_code = code
        self._payload = payload or {}

    def json(self) -> Dict[str, Any]:
        return self._payload


class FakeLatencyResponse:
    def __init__(self, latency: Any):
        self.latency = latency


class FakeProfileResponse:
    def __init__(self):
        self.profile = types.SimpleNamespace(t_comm=0.0)


class FakeLatencyRequest:
    def __init__(self, *, devices, thunderbolts, payload_sizes):
        self.devices = devices
        self.thunderbolts = thunderbolts
        self.payload_sizes = payload_sizes

    def model_dump(self) -> Dict[str, Any]:
        from .discovery import _to_json_devices

        return {
            "devices": _to_json_devices(self.devices),
            "thunderbolts": self.thunderbolts,
            "payload_sizes": self.payload_sizes,
        }


class FakeProfileRequest:
    def __init__(self, *, repo_id, thunderbolts, payload_sizes, max_batch_exp, devices):
        self.repo_id = repo_id
        self.thunderbolts = thunderbolts
        self.payload_sizes = payload_sizes
        self.max_batch_exp = max_batch_exp
        self.devices = devices

    def model_dump(self) -> Dict[str, Any]:
        from .discovery import _to_json_devices

        return {
            "repo_id": self.repo_id,
            "thunderbolts": self.thunderbolts,
            "payload_sizes": self.payload_sizes,
            "max_batch_exp": self.max_batch_exp,
            "devices": _to_json_devices(self.devices),
        }


class FakeDetokenizer:
    def __init__(self):
        self.text = ""

    def reset(self):
        self.text = ""

    def add_token(self, token: int):
        self.text += f"t{int(token)}"

    def finalize(self):
        pass


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self.chat_template = None
        self.detokenizer = FakeDetokenizer()

    def encode(self, s: Any, add_special_tokens: bool = False):
        return list(range(len(str(s))))


class FakeTokenizerWithTemplate(FakeTokenizer):
    def __init__(self):
        super().__init__()
        self.chat_template = "{{ messages }}"
        self.applied_with = None

    def apply_chat_template(self, message_dicts, add_generation_prompt, tokenize):
        self.applied_with = {
            "message_dicts": tuple((m["role"], m["content"]) for m in message_dicts),
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
        }
        return "<applied-chat-template>"


class FakeTokenResult:
    def __init__(
        self,
        token_id: int,
        logprob: float = 0.0,
        top_logprobs: Dict[int, float] | None = None,
    ):
        self.token_id = int(token_id)
        self.logprob = float(logprob)
        self.top_logprobs = top_logprobs or {}


class FakeStrategyAdapter:
    """Minimal API strategy adapter stub used by InferenceManager tests."""

    def __init__(self):
        self.connected: tuple[str, int] | None = None
        self.reset: bool = False
        self.sent: list[Dict[str, Any]] = []
        self._q: Dict[str, list[FakeTokenResult]] = {}
        self.resolved: Dict[str, Any] = {}

    async def connect_first_shard(self, ip: str, port: int):
        self.connected = (ip, port)

    async def reset_cache(self):
        self.reset = True

    async def send_tokens(self, **kwargs):
        self.sent.append(kwargs)

    async def await_token(
        self, nonce: str, timeout_s: float = 300.0
    ) -> FakeTokenResult:
        q = self._q.get(nonce)
        if not q and self._q:
            q = next(iter(self._q.values()))
        if not q:
            raise RuntimeError("no token queued")
        return q.pop(0)

    def queue_token(self, nonce: str, result: FakeTokenResult) -> None:
        self._q.setdefault(nonce, []).append(result)

    def resolve_token(self, nonce: str, result: Any) -> None:
        self.resolved[nonce] = result


class FakeModelManager:
    """Minimal model manager stub for load/unload requests."""

    def __init__(
        self,
        tok=None,
        *,
        current_model_id=None,
        load_success: bool = True,
        unload_success: bool = True,
    ):
        self.tokenizer = tok
        self.current_model_id = current_model_id
        self.load_success = load_success
        self.unload_success = unload_success
        self.load_calls: list = []
        self.unload_calls: list = []

    def is_model_available(self, model_id) -> bool:
        if model_id == "m":
            return True
        return False

    async def load_model(
        self, topology, api_properties, grpc_port, api_callback_address: str
    ):
        from dnet.api.models import APILoadModelResponse

        self.load_calls.append(
            (topology, api_properties, grpc_port, api_callback_address)
        )
        if self.load_success:
            self.current_model_id = topology.model or "m"
        return APILoadModelResponse(
            model=topology.model or "m", success=self.load_success, shard_statuses=[]
        )

    async def unload_model(self, shards):
        from dnet.api.models import UnloadModelResponse

        self.unload_calls.append(shards)
        if self.unload_success:
            self.current_model_id = None
            self.tokenizer = None
        return UnloadModelResponse(
            success=self.unload_success, shard_statuses=[], message="ok"
        )


class FakeModelProfile:
    """Simple model profile object with to_model_profile()."""

    def to_model_profile(self) -> Dict[str, Any]:
        return {"ok": True}


class FakeInferenceManager:
    """Inference manager stub used by API server tests."""

    def __init__(self, grpc_port: int = 12345):
        self.grpc_port = grpc_port
        self.connected: tuple[str, int, str] | None = None
        self.calls: list = []
        self.last: tuple | None = None

    def resolve_request(self, *a, **k):
        self.last = (a, k)

    async def connect_to_ring(self, ip: str, port: int, api_ip: str):
        self.connected = (ip, port, api_ip)

    async def generate_stream(self, req):
        import time
        from dnet.api.models import (
            ChatResponseModel,
            ChatChoice,
            ChatMessage,
        )

        yield ChatResponseModel(
            id="s1",
            choices=[
                ChatChoice(index=0, delta=ChatMessage(role="assistant", content=""))
            ],
            created=int(time.time()),
            model=req.model,
        )
        yield ChatResponseModel(
            id="s1",
            choices=[
                ChatChoice(
                    index=0, delta=ChatMessage(role="assistant", content="hello")
                )
            ],
            created=int(time.time()),
            model=req.model,
        )

    async def chat_completions(self, req):
        """Non-streaming chat completion stub returning a simple message."""
        import time
        from dnet.api.models import (
            ChatResponseModel,
            ChatChoice,
            ChatMessage,
            ChatCompletionReason,
        )

        return ChatResponseModel(
            id="c1",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="ok"),
                    finish_reason=ChatCompletionReason.STOP,
                )
            ],
            created=int(time.time()),
            model=req.model,
        )


class FakeClusterManager:
    """Minimal ClusterManager-like stub used by API HTTP server tests.

    Holds a discovery object and a simple shards mapping.
    """

    def __init__(self, shards: dict | None = None):
        from .discovery import FakeDiscovery

        self.discovery = FakeDiscovery(shards or {})
        self.shards = shards or {}
        self.current_topology = None

    async def scan_devices(self):
        return self.shards

    async def profile_cluster(
        self, model_id, embedding_size, max_batch_exp, batch_sizes
    ):
        return {}

    async def solve_topology(
        self, profiles, model_profile, model_name, num_layers, kv_bits
    ):
        return {}
