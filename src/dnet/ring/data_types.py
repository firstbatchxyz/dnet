from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple, Optional

import mlx.core as mx

from ..protos.dnet_ring_pb2 import Activation, ActivationRequest


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
    rx_enq_perf_t: float = 0.0
    tx_enq_perf_t: float = 0.0
    rx_ingress_t: float = 0.0
    rx_inflight_t: float = 0.0
    ex_enq_t: float = 0.0
    tx_enq_t: float = 0.0
    # Final token path (end-shard sampling)
    is_final: bool = False
    token_id: int = -1

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
            ),
            timestamp=self.timestamp,
            node_origin=self.node_origin,
            callback_url=self.callback_url,
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
