"""Shard models for dnet ring topology endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dnet_p2p.thunderbolt import ThunderboltConnection
from dnet_p2p import DnetDeviceProperties

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
    """Response from device profiling and latency measurement."""

    profile: Dict[str, Any] = Field(..., description="Device profile information")
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
    instance: Optional[str] = Field(
        default=None, description="Short shard instance name (service label)"
    )
