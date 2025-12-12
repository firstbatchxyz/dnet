"""Tests for static peer discovery (hostfile-based discovery)."""

import pytest
import json
from pathlib import Path

from dnet_p2p import StaticDiscovery, load_hostfile


class TestLoadHostfile:
    """Tests for load_hostfile function."""

    def test_ssh_style_single_entry(self, tmp_path: Path):
        """Load a simple SSH-style hostfile with one entry."""
        hostfile = tmp_path / "hostfile"
        hostfile.write_text("shard-1 10.0.1.100 8081 58081\n")

        devices = load_hostfile(hostfile)

        assert len(devices) == 1
        assert "shard-1" in devices
        device = devices["shard-1"]
        assert device.instance == "shard-1"
        assert device.local_ip == "10.0.1.100"
        assert device.server_port == 8081
        assert device.shard_port == 58081
        assert device.is_manager is False
        assert device.is_busy is False

    def test_ssh_style_multiple_entries(self, tmp_path: Path):
        """Load SSH-style hostfile with multiple entries."""
        hostfile = tmp_path / "hostfile"
        hostfile.write_text(
            "shard-1 10.0.1.100 8081 58081\n"
            "shard-2 10.0.1.101 8082 58082\n"
            "shard-3 10.0.1.102 8083 58083\n"
        )

        devices = load_hostfile(hostfile)

        assert len(devices) == 3
        assert devices["shard-1"].local_ip == "10.0.1.100"
        assert devices["shard-2"].local_ip == "10.0.1.101"
        assert devices["shard-3"].local_ip == "10.0.1.102"

    def test_ssh_style_with_comments(self, tmp_path: Path):
        """Comments and blank lines are ignored."""
        hostfile = tmp_path / "hostfile"
        hostfile.write_text(
            "# This is a comment\n"
            "\n"
            "shard-1 10.0.1.100 8081 58081\n"
            "  # Indented comment\n"
            "\n"
            "shard-2 10.0.1.101 8082 58082\n"
        )

        devices = load_hostfile(hostfile)

        assert len(devices) == 2
        assert "shard-1" in devices
        assert "shard-2" in devices

    def test_json_format(self, tmp_path: Path):
        """Load JSON-formatted hostfile."""
        hostfile = tmp_path / "hostfile.json"
        data = [
            {
                "instance": "shard-1",
                "local_ip": "10.0.1.100",
                "server_port": 8081,
                "shard_port": 58081,
            },
            {
                "instance": "shard-2",
                "local_ip": "10.0.1.101",
                "server_port": 8082,
                "shard_port": 58082,
            },
        ]
        hostfile.write_text(json.dumps(data))

        devices = load_hostfile(hostfile)

        assert len(devices) == 2
        assert devices["shard-1"].local_ip == "10.0.1.100"
        assert devices["shard-2"].server_port == 8082

    def test_file_not_found(self):
        """Raise FileNotFoundError for missing hostfile."""
        with pytest.raises(FileNotFoundError):
            load_hostfile("/nonexistent/path/hostfile")

    def test_invalid_ssh_format(self, tmp_path: Path):
        """Raise ValueError for invalid SSH-style format."""
        hostfile = tmp_path / "hostfile"
        hostfile.write_text("shard-1 10.0.1.100\n")  # Missing ports

        with pytest.raises(ValueError, match="Invalid hostfile format"):
            load_hostfile(hostfile)

    def test_invalid_port_number(self, tmp_path: Path):
        """Raise ValueError for non-numeric port."""
        hostfile = tmp_path / "hostfile"
        hostfile.write_text("shard-1 10.0.1.100 abc 58081\n")

        with pytest.raises(ValueError, match="Invalid port number"):
            load_hostfile(hostfile)

    def test_invalid_json(self, tmp_path: Path):
        """Raise ValueError for invalid JSON."""
        hostfile = tmp_path / "hostfile"
        hostfile.write_text("[{invalid json}]")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_hostfile(hostfile)

    def test_accepts_string_path(self, tmp_path: Path):
        """Accept string path in addition to Path object."""
        hostfile = tmp_path / "hostfile"
        hostfile.write_text("shard-1 10.0.1.100 8081 58081\n")

        devices = load_hostfile(str(hostfile))

        assert len(devices) == 1


class TestStaticDiscovery:
    """Tests for StaticDiscovery class."""

    @pytest.fixture
    def hostfile(self, tmp_path: Path) -> Path:
        """Create a test hostfile."""
        hf = tmp_path / "hostfile"
        hf.write_text("shard-1 10.0.1.100 8081 58081\nshard-2 10.0.1.101 8082 58082\n")
        return hf

    def test_init(self, hostfile: Path):
        """StaticDiscovery initializes correctly."""
        discovery = StaticDiscovery(
            hostfile=hostfile,
            own_instance="api-node",
            own_http_port=8080,
            own_grpc_port=58080,
        )

        assert discovery.instance_name() == "api-node"
        assert discovery.is_created() is True

    def test_instance_name(self, hostfile: Path):
        """instance_name returns the configured instance name."""
        discovery = StaticDiscovery(
            hostfile=hostfile,
            own_instance="my-api",
            own_http_port=8080,
            own_grpc_port=58080,
        )

        assert discovery.instance_name() == "my-api"

    def test_async_start_stop(self, hostfile: Path):
        """async_start and async_stop work correctly."""
        import asyncio

        async def run_test():
            discovery = StaticDiscovery(
                hostfile=hostfile,
                own_instance="api",
                own_http_port=8080,
                own_grpc_port=58080,
            )

            assert discovery.is_running() is False

            await discovery.async_start()
            assert discovery.is_running() is True

            result = await discovery.async_stop()
            assert result == 0
            assert discovery.is_running() is False

        asyncio.run(run_test())

    def test_async_get_properties(self, hostfile: Path):
        """async_get_properties returns static peers."""
        import asyncio

        async def run_test():
            discovery = StaticDiscovery(
                hostfile=hostfile,
                own_instance="api",
                own_http_port=8080,
                own_grpc_port=58080,
            )

            devices = await discovery.async_get_properties()

            assert len(devices) == 2
            assert "shard-1" in devices
            assert "shard-2" in devices
            assert devices["shard-1"].local_ip == "10.0.1.100"

        asyncio.run(run_test())

    def test_async_get_own_properties(self, hostfile: Path):
        """async_get_own_properties returns own properties."""
        import asyncio

        async def run_test():
            discovery = StaticDiscovery(
                hostfile=hostfile,
                own_instance="api-node",
                own_http_port=8080,
                own_grpc_port=58080,
                own_ip="192.168.1.100",
            )

            props = await discovery.async_get_own_properties()

            assert props.instance == "api-node"
            assert props.local_ip == "192.168.1.100"
            assert props.server_port == 8080
            assert props.shard_port == 58080
            assert props.is_manager is True

        asyncio.run(run_test())

    def test_async_set_is_busy(self, hostfile: Path):
        """async_set_is_busy updates busy status."""
        import asyncio

        async def run_test():
            discovery = StaticDiscovery(
                hostfile=hostfile,
                own_instance="api",
                own_http_port=8080,
                own_grpc_port=58080,
            )

            props = await discovery.async_get_own_properties()
            assert props.is_busy is False

            await discovery.async_set_is_busy(True)

            props = await discovery.async_get_own_properties()
            assert props.is_busy is True

        asyncio.run(run_test())

    def test_sync_get_properties(self, hostfile: Path):
        """Synchronous get_properties works."""
        discovery = StaticDiscovery(
            hostfile=hostfile,
            own_instance="api",
            own_http_port=8080,
            own_grpc_port=58080,
        )

        devices = discovery.get_properties()

        assert len(devices) == 2

    def test_sync_start_stop(self, hostfile: Path):
        """Synchronous start/stop works."""
        discovery = StaticDiscovery(
            hostfile=hostfile,
            own_instance="api",
            own_http_port=8080,
            own_grpc_port=58080,
        )

        discovery.start()
        assert discovery.is_running() is True

        discovery.stop()
        assert discovery.is_running() is False

    def test_create_instance_noop(self, hostfile: Path):
        """create_instance is a no-op but doesn't raise."""
        discovery = StaticDiscovery(
            hostfile=hostfile,
            own_instance="api",
            own_http_port=8080,
            own_grpc_port=58080,
        )

        # Should not raise
        discovery.create_instance("other", 9000, 59000, is_manager=False)
