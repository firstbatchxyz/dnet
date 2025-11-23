"""Discovery/device property fakes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import types


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


@dataclass
class FakeProps:
    """Minimal device properties used by tests and fake discovery."""

    instance: str
    local_ip: str
    server_port: int
    is_manager: bool = False

    def model_dump(self) -> Dict[str, Any]:
        return {
            "instance": self.instance,
            "local_ip": self.local_ip,
            "server_port": self.server_port,
            "is_manager": self.is_manager,
        }


class FakeTBConn:
    """Minimal Thunderbolt connection holder (ip + originating instance)."""

    def __init__(self, ip_addr: str, instance: str | None = None):
        self.ip_addr = ip_addr
        self.instance = instance or "S2"


class FakeDiscovery:
    """Minimal discovery stub returning static devices and "self" props."""

    def __init__(self, shards: Dict[str, Any] | None = None):
        self._shards = shards or {}

    async def async_get_properties(self) -> Dict[str, FakeProps]:
        out: Dict[str, FakeProps] = {}
        for k, v in self._shards.items():
            if isinstance(v, dict):
                d = dict(v)
                d.setdefault("instance", k)
                d.setdefault("local_ip", "0.0.0.0")
                d.setdefault("server_port", 0)
                d.setdefault("is_manager", False)
                out[k] = FakeProps(
                    d["instance"], d["local_ip"], d["server_port"], d["is_manager"]
                )
            else:
                out[k] = v
        return out

    async def async_get_own_properties(self):
        # Minimal self-properties object
        return types.SimpleNamespace(
            instance="self",
            local_ip="127.0.0.1",
            server_port=0,
            is_manager=True,
            thunderbolt=None,
        )

    def instance_name(self) -> str:
        return "self"
