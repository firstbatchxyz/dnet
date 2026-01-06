"""Core message DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Tuple, Optional

import mlx.core as mx

from dnet.protos.dnet_ring_pb2 import Activation, ActivationRequest


# Internal dataclass for easier handling
@dataclass(slots=True)
class ActivationMessage:
    nonce: str
    pool_id: int
    batch_size: int
    shape: Tuple[int, ...]
    dtype: str
    layer_id: int
    timestamp: int
    node_origin: str
    callback_url: str
    # Optional direct tensor reference to avoid staging/copies
    tensor: Optional[mx.array] = None
    # Local profiling timestamps (perf_counter seconds)
    recv_perf_t: float = 0.0
    enq_perf_t: float = 0.0
    # TX queue enqueue time (perf_counter seconds)
    tx_enq_perf_t: float = 0.0
    # Final token path (end-shard sampling)
    is_final: bool = False
    token_id: int = -1
    logprob: float = 0.0
    top_logprobs: Optional[dict[int, float]] = None

    # Request control
    req_logprobs: bool = False
    req_top_logprobs: int = 0
    # Decoding parameters
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    # CP RoPE offset
    rope_offset: int = 0

    @classmethod
    def from_proto(cls, proto_msg: ActivationRequest, pool_id: int = 0):
        """Create from protobuf message"""
        return cls(
            nonce=proto_msg.nonce,
            pool_id=pool_id,
            batch_size=proto_msg.activation.batch_size,
            shape=tuple(proto_msg.activation.shape),
            dtype=proto_msg.activation.dtype,
            layer_id=proto_msg.activation.layer_id,
            timestamp=proto_msg.timestamp,
            node_origin=proto_msg.node_origin,
            callback_url=proto_msg.callback_url,
            req_logprobs=proto_msg.logprobs,
            req_top_logprobs=proto_msg.top_logprobs,
            temperature=proto_msg.temperature
            if proto_msg.HasField("temperature")
            else 1.0,
            top_p=proto_msg.top_p if proto_msg.HasField("top_p") else 1.0,
            top_k=proto_msg.top_k if proto_msg.HasField("top_k") else -1,
            repetition_penalty=proto_msg.repetition_penalty
            if proto_msg.HasField("repetition_penalty")
            else 1.0,
            min_p=proto_msg.min_p if proto_msg.HasField("min_p") else 0.0,
            min_tokens_to_keep=proto_msg.min_tokens_to_keep
            if proto_msg.HasField("min_tokens_to_keep")
            else 1,
            rope_offset=proto_msg.activation.rope_offset,
        )

    def to_proto(self, data: bytes) -> ActivationRequest:
        """Convert to protobuf request"""
        return ActivationRequest(
            nonce=self.nonce,
            activation=Activation(
                data=data,
                batch_size=self.batch_size,
                shape=list(self.shape),
                layer_id=self.layer_id,
                dtype=self.dtype,
                rope_offset=self.rope_offset,
            ),
            timestamp=self.timestamp,
            node_origin=self.node_origin,
            callback_url=self.callback_url,
            logprobs=self.req_logprobs,
            top_logprobs=self.req_top_logprobs,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            min_p=self.min_p,
            min_tokens_to_keep=self.min_tokens_to_keep,
        )


@dataclass(slots=True)
class WeightRequest:
    weight_id: str
    layer_id: int
    priority: int = 0


class PoolStatus(str, Enum):
    FREE = "free"
    ALLOCATED = "allocated"
    IN_USE = "in_use"


class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int


@dataclass
class TokenResult:
    token_id: int
    logprob: float = 0.0
    top_logprobs: dict[int, float] = field(default_factory=dict)


__all__ = [
    "ActivationMessage",
    "WeightRequest",
    "PoolStatus",
    "StopCondition",
    "TokenResult",
]
