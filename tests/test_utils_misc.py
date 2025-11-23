"""Tests: small pure utility functions in utils.latency and utils.network."""

import pytest

from dnet.utils.latency import (
    LatencyResults,
    DeviceLatencyResult,
    LatencyMeasurement,
    calculate_median_latency_seconds,
)
from dnet.utils.network import parse_ipaddress


pytestmark = [pytest.mark.core]


# two devices, three measurements total -> median of [10, 20, 30] ms -> 20 ms -> 0.02 s
def test_calculate_median_latency_seconds():
    results = LatencyResults(
        results={
            "A": DeviceLatencyResult(
                target_node_id="A",
                measurements=[
                    LatencyMeasurement(payload_size=1, latency_ms=10.0, success=True),
                    LatencyMeasurement(payload_size=2, latency_ms=20.0, success=True),
                ],
            ),
            "B": DeviceLatencyResult(
                target_node_id="B",
                measurements=[
                    LatencyMeasurement(payload_size=1, latency_ms=30.0, success=True)
                ],
            ),
        }
    )
    m = calculate_median_latency_seconds(results)
    assert m is not None and abs(m - 0.020) < 1e-6


def test_parse_ipaddress_ipv4_and_hostname():
    a = parse_ipaddress("127.0.0.1:8080")
    assert a.host == "127.0.0.1" and a.port == 8080 and a.format() == "127.0.0.1:8080"
    b = parse_ipaddress("example.com:443")
    assert b.host == "example.com" and b.port == 443 and b.format() == "example.com:443"

    with pytest.raises(Exception):
        parse_ipaddress("bad host:80")
