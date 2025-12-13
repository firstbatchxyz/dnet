"""Discovery/device property fakes."""

from __future__ import annotations

from typing import Any, Dict

from dnet_p2p import DnetDeviceProperties


def _to_json_devices(devices: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in devices.items():
        if isinstance(v, dict):
            d = v
        else:
            d = {
                "instance": getattr(v, "instance", k),
                "local_ip": getattr(v, "local_ip", "0.0.0.0"),
                "server_port": getattr(v, "server_port", 0),
                "is_manager": getattr(v, "is_manager", False),
            }
        out[k] = d
    return out


def FakeProps(
    instance: str, local_ip: str, server_port: int, *, is_manager: bool = False
) -> DnetDeviceProperties:
    """Strict fake device properties that match production type (`DnetDeviceProperties`)."""

    return DnetDeviceProperties(
        is_manager=is_manager,
        is_busy=False,
        instance=instance,
        server_port=server_port,
        shard_port=int(server_port) + 1000,
        local_ip=local_ip,
        thunderbolt=None,
    )


class FakeTBConn:
    """Minimal Thunderbolt connection holder (ip + originating instance)."""

    def __init__(self, ip_addr: str, instance: str | None = None):
        self.ip_addr = ip_addr
        self.instance = instance or "S2"


class FakeDiscovery:
    """Minimal discovery stub returning static devices and "self" props."""

    def __init__(self, shards: Dict[str, Any] | None = None):
        self._shards = shards or {}

    async def async_get_properties(self) -> Dict[str, DnetDeviceProperties]:
        out: Dict[str, DnetDeviceProperties] = {}
        for k, v in self._shards.items():
            if isinstance(v, dict):
                d = dict(v)
                d.setdefault("instance", k)
                d.setdefault("local_ip", "0.0.0.0")
                d.setdefault("server_port", 0)
                d.setdefault("is_manager", False)
                d.setdefault("shard_port", int(d["server_port"]) + 1000)
                d.setdefault("is_busy", False)
                d.setdefault("thunderbolt", None)
                out[k] = DnetDeviceProperties(**d)
            else:
                out[k] = v
        return out

    async def async_get_own_properties(self):
        return DnetDeviceProperties(
            is_manager=True,
            is_busy=False,
            instance="self",
            server_port=0,
            shard_port=0,
            local_ip="127.0.0.1",
            thunderbolt=None,
        )

    def instance_name(self) -> str:
        return "self"
