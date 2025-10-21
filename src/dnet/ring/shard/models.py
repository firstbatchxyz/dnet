"""Shard models for dnet ring topology endpoints."""

from typing import Any, Dict, List, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from dnet_p2p import DnetDeviceProperties, ThunderboltConnection

from dnet.utils.latency import LatencyResults


class ShardLoadModelRequest(BaseModel):
    """Request to load model with specified layers on shard."""

    model_path: str = Field(..., description="Model path or HuggingFace repo ID")
    total_layers: int = Field(..., description="Total number of layers in the model")
    layers: List[int] = Field(..., description="Layer indices to load on this shard")
    warmup: bool = Field(
        default=False, description="Whether to perform warmup after loading"
    )
    next_node: Optional[DnetDeviceProperties] = Field(
        default=None, description="Next shard in the ring"
    )
    window_size: int = Field(..., description="Window size (computed from k)")
    residency_size: int = Field(
        ..., description="Resident layers (n) allowed on GPU at once"
    )
    kv_bits: Literal["4bit", "8bit", "fp16"] = Field(
        ..., description="KV cache quantization ('4bit'|'8bit'|'fp16')"
    )
    api_callback_address: str = Field(
        ...,
        description="API callback address for final layer completion (gRPC host:port)",
    )


class ShardLoadModelResponse(BaseModel):
    """Response from model loading operation on shard."""

    success: bool = Field(..., description="Whether loading succeeded")
    message: str = Field(..., description="Status or error message")
    layers_loaded: List[int] = Field(..., description="Layers successfully loaded")
    load_time_ms: float = Field(
        ..., description="Time taken to load model in milliseconds"
    )


class ShardUnloadModelResponse(BaseModel):
    """Response from model unloading operation on shard."""

    success: bool = Field(..., description="Whether unloading succeeded")
    message: str = Field(..., description="Status or error message")


class ShardProfileRequest(BaseModel):
    """Request to profile device and measure latencies."""

    api_address: Optional[str] = Field( ..., description="API Address" ) 
    devices: Dict[str, DnetDeviceProperties] = Field(
        ..., description="Device information mapping"
    )
    payload_sizes: List[int] = Field(
        default=[1024], description="Payload sizes to test for latency measurement"
    )
    max_batch_exp: int = Field(
        default=2, description="Maximum batch size exponent (2^max_batch_exp)"
    )
    repo_id: str = Field(..., description="Model repository ID for device profiling")
    thunderbolts: Dict[str, ThunderboltConnection] = Field(
        default={}, description="Thunderbolt connection information from this device"
    )


class ShardProfileResponse(BaseModel):
    """Response from device profiling."""

    profile: Dict[str, Any] = Field(..., description="Device profile information")


class MeasureLatencyRequest(BaseModel):
    """Request to measure latency to other devices."""

    devices: Dict[str, DnetDeviceProperties] = Field(
        ..., description="Device information mapping"
    )
    thunderbolts: Dict[str, ThunderboltConnection] = Field(
        default={}, description="Thunderbolt connection information from this device"
    )
    payload_sizes: List[int] = Field(
        default=[1024], description="Payload sizes to test for latency measurement"
    )


class MeasureLatencyResponse(BaseModel):
    """Response from latency measurement."""

    latency: LatencyResults = Field(..., description="Latency measurement results")


class HealthResponse(BaseModel):
    """Response from health check endpoint."""

    status: str = Field(..., description="Health status (e.g., 'ok')")
    node_id: int = Field(..., description="Node identifier")
    running: bool = Field(..., description="Whether the node is running")
    model_loaded: bool = Field(..., description="Whether a model is currently loaded")
    model_path: Optional[str] = Field(
        None, description="Path to currently loaded model"
    )
    assigned_layers: List[int] = Field(..., description="Layers assigned to this shard")
    queue_size: int = Field(..., description="Current activation queue size")
    grpc_port: int = Field(..., description="gRPC server port")
    http_port: int = Field(..., description="HTTP server port")
    instance: Optional[str] = Field(default=None, description="Shard name")


# Tracer 

class TraceConfigRequest(BaseModel):
    file: str = Field(..., description="File for trace streaming")
    streaming: bool = Field(..., description="Toggle for trace streaming to file")
    include_prefixes: List[str] = Field(default=("src/dnet/"), description="")
    include_c_calls: bool = Field(default=False, description="")
    budget: int = Field(default=0, description="Max amount of traces in memory")
    enabled: bool = Field(default=False, description="Start or stop tracing")
    node_id: Optional[str] = Field(default="NONE", description="")
    record_pid_tid: bool = Field(default=True, descriptino="") 
    aggregate: bool = Field(default=True, description="")
    aggregate_url: Optional[str] = Field(default=None, description="")
    agg_max_events: int = Field(default=300, description="Threshold for sending frames to API")

class TraceConfigResponse(BaseModel):
    ok: bool = True
    
class TraceEvent(BaseModel):
    type: str = Field(..., description="Event type/phase")
    name: str = Field(..., description="Span/mark name")
    ts: float = Field(..., description="Timestamp in microseconds")
    args: Dict[str, Any] = Field(default_factory=dict)
    req_id: Optional[str] = None
    pid: Optional[int] = None
    tid: Optional[int] = None

class TraceIngestBatch(BaseModel):
    run_id: str = Field(..., description="Bench run identifier")
    node_id: str = Field(..., description="Shard/service identity")
    events: List[TraceEvent] = Field(default_factory=list)
    #dropped: Optional[int] = Field(default=0, description="Events dropped on sender")
    #max_ts: Optional[int] = Field(default=None, description="Max ts_us in this batch")
    #last: Optional[bool] = Field(default=False, description="Sender indicates end-of-run")
    #schema_version: int = Field(default=1)

class TraceIngestResponse(BaseModel):
    ok: bool = True
    accepted: int = 0
    message: Optional[str] = None
