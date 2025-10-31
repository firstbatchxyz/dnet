from typing import Any, Dict, List, Optional, Literal
from dnet_p2p import DnetDeviceProperties
from pydantic import BaseModel, Field


class LayerAssignment(BaseModel):
    """Layer assignment for a single device in ring topology."""

    instance: str = Field(..., description="Target device name")
    layers: List[List[int]] = Field(
        ..., description="Layer indices per round (k sublists)"
    )
    next_instance: Optional[str] = Field(
        None, description="Next device name in ring (null if connects to API)"
    )
    window_size: int = Field(..., description="Window size for this device")
    residency_size: int = Field(
        ..., description="Number of resident layers on GPU any given time"
    )


class TopologyInfo(BaseModel):
    """Stored topology information for the current model."""

    model: Optional[str] = Field(
        ..., description="Loaded model name or HuggingFace repo ID"
    kv_bits: Literal["4bit", "8bit", "fp16"] = Field(
        ..., description="KV cache quantization used by solver and shards"
    )
    num_layers: int = Field(..., description="Total number of layers in model")
    devices: List[DnetDeviceProperties] = Field(
        ..., description="Devices (in solver order)"
    )
    assignments: List[LayerAssignment] = Field(
        ..., description="Layer assignments per device"
    )
    solution: Dict[str, Any] = Field(..., description="Solver result details")
