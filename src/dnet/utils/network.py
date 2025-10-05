"""Network-related utilities for dnet."""

import argparse
import ipaddress
import re
from dataclasses import dataclass


# RFC 1035
HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*$"
)


@dataclass(frozen=True, slots=True)
class NodeAddress:
    """Represents a network node address (host:port)."""

    host: str
    port: int

    def format(self) -> str:
        """Format address as 'host:port' string.

        Returns:
            Formatted address string
        """
        return f"{self.host}:{self.port}"


def parse_ipaddress(value: str) -> NodeAddress:
    """Parse a host:port string into NodeAddress.

    Args:
        value: String in format 'host:port' (e.g., '192.168.1.1:8080')

    Returns:
        NodeAddress instance

    Raises:
        argparse.ArgumentTypeError: If the value is invalid
    """
    try:
        host, port_str = value.rsplit(":", 1)
        port = int(port_str)
        if not (0 <= port <= 65535):
            raise ValueError("port out of range")

        # validate IP
        try:
            ipaddress.ip_address(host)
        except ValueError:
            # not an IP → check hostname
            if not HOSTNAME_RE.match(host):
                raise ValueError("invalid hostname")

        return NodeAddress(host, port)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid host:port '{value}': {e}")
