"""
Shared utilities for Context Parallelism scripts.

Common functionality for prepare_cp_model.py and stress_test_cp.py.
"""

from functools import lru_cache
from typing import Literal

import requests
from dnet_p2p import DnetDeviceProperties

from dnet.api.models import ManualDevice
from dnet.config import DnetSettings


@lru_cache(maxsize=1)
def _fetch_settings(api_url: str) -> DnetSettings | None:
    """Fetch and cache settings from API as typed DnetSettings."""
    try:
        response = requests.get(f"{api_url}/v1/settings", timeout=5)
        if response.status_code == 200:
            return DnetSettings.model_validate(response.json())
    except (requests.RequestException, Exception):
        pass
    return None


def get_kv_bits_from_server(api_url: str) -> Literal["4bit", "8bit", "fp16"]:
    """Get kv_bits from server settings via API."""
    settings = _fetch_settings(api_url)
    if settings:
        mode = settings.kv_cache.mode
        if mode in ("4bit", "8bit", "fp16"):
            return mode  # type: ignore
    return "8bit"


def get_devices(api_url: str) -> dict[str, DnetDeviceProperties]:
    """Fetch available devices from API. Returns {instance: DnetDeviceProperties}."""
    response = requests.get(f"{api_url}/v1/devices")
    response.raise_for_status()
    data = response.json()
    devices_raw = data.get("devices", {})
    return {
        instance: DnetDeviceProperties(**props)
        for instance, props in devices_raw.items()
    }


def get_shards(api_url: str) -> list[ManualDevice]:
    """Get shard devices (non-managers) as ManualDevice list."""
    devices = get_devices(api_url)
    shards = []
    for instance, props in devices.items():
        if props.is_manager:
            continue
        shards.append(
            ManualDevice(
                instance=instance,
                local_ip=props.local_ip,
                server_port=props.server_port,
                shard_port=props.shard_port,
            )
        )
    return shards


def get_topology(api_url: str) -> dict | None:
    """Fetch current topology from API. Returns None if not set."""
    try:
        response = requests.get(f"{api_url}/v1/topology")
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


def get_api_settings(api_url: str) -> DnetSettings | None:
    """Fetch settings from API as typed DnetSettings.

    Note: Uses cached _fetch_settings internally.
    """
    return _fetch_settings(api_url)


def is_cp_enabled(api_url: str) -> bool:
    """Check if context parallelism is enabled on the API server."""
    settings = _fetch_settings(api_url)
    if settings:
        return settings.context_parallel.enabled
    return False


def get_recommended_test_sizes(num_shards: int) -> list[int]:
    """Get recommended context sizes for CP testing based on shard count.

    Based on design doc memory table:
    - Single device (24GB): ~32K comfortable, 128K tight
    - 2 devices: can handle 128K+ distributed
    - 4 devices: can handle 256K+ distributed

    Returns context lengths that should stress-test CP properly.
    """
    if num_shards <= 1:
        # Single device - test up to comfortable limit
        return [1000, 4000, 8000, 16000, 32000]
    elif num_shards == 2:
        # 2 shards - test beyond single-device capacity
        return [8000, 16000, 32000, 48000, 64000, 96000]
    else:
        # 3+ shards - test long contexts
        return [16000, 32000, 64000, 96000, 128000]


# Context length thresholds (from design doc)
SINGLE_DEVICE_COMFORTABLE = 32000  # ~4GB KV cache
SINGLE_DEVICE_TIGHT = 128000  # ~16GB KV cache
CP_MIN_BENEFIT_THRESHOLD = 32000  # Below this, CP overhead may not be worth it
