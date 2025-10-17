from typing import Any, Dict, List, Optional
from dnet_p2p import DnetDeviceProperties
from pydantic import BaseModel, Field


class LayerAssignment(BaseModel):
    """Layer assignment for a single device in ring topology."""

    service: str = Field(..., description="Target device service name")
    layers: List[List[int]] = Field(
        ..., description="Layer indices per round (k sublists)"
    )
    next_service: Optional[str] = Field(
        None, description="Next node service name in ring (null if connects to API)"
    )
    window_size: int = Field(..., description="Window size for this device")


class TopologyInfo(BaseModel):
    """Stored topology information for the current model."""

    model: str = Field(..., description="Model name or HuggingFace repo ID")
    num_layers: int = Field(..., description="Total number of layers in model")
    devices: List[DnetDeviceProperties] = Field(
        ..., description="Devices (in solver order)"
    )
    assignments: List[LayerAssignment] = Field(
        ..., description="Layer assignments per device"
    )
    solution: Dict[str, Any] = Field(..., description="Solver result details")
