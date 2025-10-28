"""Latency measurement utilities and models."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LatencyMeasurement(BaseModel):
    """Individual latency measurement for a specific payload size."""

    payload_size: int = Field(..., description="Size of payload in bytes")
    latency_ms: Optional[float] = Field(
        None, description="Measured latency in milliseconds"
    )
    success: bool = Field(..., description="Whether the measurement was successful")
    error: Optional[str] = Field(
        None, description="Error message if measurement failed"
    )


class DeviceLatencyResult(BaseModel):
    """Latency measurement results for a single device."""

    target_node_id: Optional[str] = Field(None, description="ID of the target node")
    measurements: List[LatencyMeasurement] = Field(
        default_factory=list, description="List of latency measurements"
    )
    success: bool = Field(True, description="Whether any measurements succeeded")
    error: Optional[str] = Field(
        None, description="Error message if all measurements failed"
    )


class LatencyResults(BaseModel):
    """Collection of latency results for multiple devices."""

    results: Dict[str, DeviceLatencyResult] = Field(
        default_factory=dict,
        description="Mapping of shard names to their latency results",
    )


def calculate_median_latency_seconds(
    latency_results: LatencyResults,
) -> Optional[float]:
    """Calculate median latency in seconds from latency measurement results.

    Args:
        latency_results: LatencyResults object containing measurements from all devices.

    Returns:
        Median latency in seconds, or None if no valid measurements found.
    """
    all_latencies = []
    for _instance, result in latency_results.results.items():
        for measurement in result.measurements:
            if measurement.success and measurement.latency_ms is not None:
                all_latencies.append(measurement.latency_ms)

    if not all_latencies:
        return None

    all_latencies.sort()
    n = len(all_latencies)
    median_latency_ms = (
        all_latencies[n // 2]
        if n % 2 == 1
        else (all_latencies[n // 2 - 1] + all_latencies[n // 2]) / 2
    )
    return median_latency_ms / 1000.0  # convert ms to seconds
