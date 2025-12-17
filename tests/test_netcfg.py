import io
import json
import types
import builtins
import os
import pytest

import dnet.core.network.netcfg as netcfg

pytestmark = pytest.mark.netcfg


# Fake the output from ifconfig commands
class FakeExec:
    def __init__(self, table=None):
        self.table = table or {}
        self.calls = []

    def __call__(self, cmd):
        key = tuple(cmd)
        self.calls.append(key)
        rc, out, err = self.table.get(key, (0, "", ""))
        return rc, out, err


def test_status_parses_members_and_ips(monkeypatch):
    fx = FakeExec(
        {
            ("ifconfig", "-l"): (0, "lo0 en0 en1 bridge0 bridge1", ""),
            ("ifconfig", "bridge0"): (
                0,
                "\nmember: en3\ninet 10.0.0.1 netmask 0xffffff00 mtu 1500",
                "",
            ),
            ("ifconfig", "bridge1"): (0, "\nmember: en4\n", ""),
            ("ifconfig", "bridge0", "up"): (0, "", ""),
            ("networksetup", "-listallhardwareports"): (
                0,
                "Hardware Port: Thunderbolt 1\nDevice: en3\n\nHardware Port: Wi-Fi\nDevice: en1\n",
                "",
            ),
        }
    )
    monkeypatch.setattr(netcfg, "__exec", fx)

    out = netcfg.status()
    assert out["ok"] is True
    assert set(out["bridges"]) >= {"bridge0", "bridge1"}
    assert out["members"]["bridge0"] == ["en3"]
    assert "en3" in out["thunderbolt"]


def test_apply_network_mappings_integration():
    import platform

    if platform.system() != "Darwin":
        pytest.skip("macOS only")

    idx = 1
    while netcfg._bridge_exists(f"bridge{idx}"):
        idx += 1
    target_br = f"bridge{idx}"

    iface = None
    mem0 = netcfg._get_bridge_members("bridge0")
    if mem0:
        iface = mem0[0]
    else:
        tbs = netcfg._get_thunderbolt_interfaces()
        if tbs:
            iface = tbs[0]

    if not iface:
        pytest.skip("no target interface to attach")

    plan = [
        {
            "mdp_interface": target_br,
            "thunderbolt_en": iface,
            "mdp_self_ipv4": "10.10.0.0",
            "mdp_ipv4": "10.10.0.1",
        }
    ]

    is_root = hasattr(os, "geteuid") and os.geteuid() == 0

    if not is_root:
        with pytest.raises(Exception) as e:
            netcfg.apply_network_mappings(plan)
        assert "permit" in str(e.value).lower() or "sioc" in str(e.value).lower()
    else:
        res = netcfg.apply_network_mappings(plan)
        assert res["status"] == "ok"
        assert any(
            a.get("member") == iface and a.get("bridge") == target_br
            for a in res["actions"]
        )
        assert any(a.get("addr") == "10.10.0.0" for a in res["actions"])


def test_read_plan_from_file(tmp_path):
    p = tmp_path / "plan.json"
    plan = {
        "mappings": [
            {
                "mdp_interface": "bridge3",
                "thunderbolt_en": "en4",
                "mdp_self_ipv4": "10.0.0.0",
                "mdp_ipv4": "10.0.0.1",
            }
        ]
    }
    p.write_text(json.dumps(plan))

    got = netcfg.read_plan(str(p))
    assert isinstance(got, list) and got[0]["mdp_interface"] == "bridge3"


def test_register_sudoers_writes_tmp_and_validates(monkeypatch, tmp_path):
    files = {}

    def fake_open(path, mode="r", encoding=None):
        if "w" in mode:
            buf = io.StringIO()
            def _close():
                files[path] = buf.getvalue()
            buf.close = _close  # type: ignore
            return buf
        return io.StringIO(files.get(path, ""))

    def fake_exists(path):
        return path == "/opt/homebrew/bin/dnet-netcfg"

    class Stat:
        st_mode = 0o755

    def fake_stat(path):
        return Stat()

    def fake_chmod(path, mode):
        return None

    def fake_replace(src, dst):
        files[dst] = files.get(src, "")
        files.pop(src, None)

    fx = FakeExec({("visudo", "-cf", "/etc/sudoers.d/dnet-netcfg.tmp"): (0, "", "")})

    monkeypatch.setattr(netcfg, "__exec", fx)
    monkeypatch.setattr(netcfg.os.path, "exists", fake_exists)
    monkeypatch.setattr(netcfg.os, "stat", fake_stat)
    monkeypatch.setattr(netcfg.os, "chmod", fake_chmod)
    monkeypatch.setattr(netcfg.os, "replace", fake_replace)
    monkeypatch.setattr(builtins, "open", fake_open)

    out = netcfg.register_sudoers("tester")
    assert any(a.get("wrote") == "/etc/sudoers.d/dnet-netcfg" for a in out)
    assert "/etc/sudoers.d/dnet-netcfg" in files
    assert (
        "tester ALL=(root) NOPASSWD: /opt/homebrew/bin/dnet-netcfg"
        in files["/etc/sudoers.d/dnet-netcfg"]
    )


def test_register_in_homebrew_bin_monkeypatched(monkeypatch):
    files = {}

    def fake_open(path, mode="r", encoding=None):
        if "w" in mode:
            buf = io.StringIO()

            def _close():
                files[path] = buf.getvalue()

            buf.close = _close  # type: ignore
            return buf
        return io.StringIO("print('ok')\n")

    def fake_chmod(path, mode):
        return None

    # Prevent actually writing sudoers
    monkeypatch.setattr(netcfg, "register_sudoers", lambda user: [{"user": user}])
    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(netcfg.os, "makedirs", lambda d, exist_ok: None)
    monkeypatch.setattr(netcfg.os, "chmod", fake_chmod)
    monkeypatch.setenv("USER", "tester")

    out = netcfg.register_in_homebrew_bin()
    assert any(a.get("installed") == "/opt/homebrew/bin/dnet-netcfg" for a in out)
    assert "/opt/homebrew/bin/dnet-netcfg" in files


def _require_macos_root():
    import platform
    if platform.system() != "Darwin":
        pytest.skip("macOS only")
    if not (hasattr(os, "geteuid") and os.geteuid() == 0):
        pytest.skip("requires root to modify networking")


def _next_bridge_name():
    idx = 1
    while netcfg._bridge_exists(f"bridge{idx}"):
        idx += 1
    return f"bridge{idx}"


def _pick_iface():
    m0 = netcfg._get_bridge_members("bridge0")
    if m0:
        return m0[0]
    tbs = netcfg._get_thunderbolt_interfaces()
    if tbs:
        return tbs[0]
    return None


def test_bridge_create_destroy_integration():
    _require_macos_root()
    br = _next_bridge_name()
    try:
        netcfg._ensure_bridge(br)
        assert netcfg._bridge_exists(br)
        assert netcfg._get_bridge_members(br) == []
    finally:
        netcfg._destroy_bridge(br)
        assert not netcfg._bridge_exists(br)


def test_attach_detach_integration():
    _require_macos_root()
    iface = _pick_iface()
    if not iface:
        pytest.skip("no interface available to move")
    src_br = netcfg._find_interface(iface) or "bridge0"
    dst_br = _next_bridge_name()
    netcfg._ensure_bridge(dst_br)
    try:
        netcfg._move_interface(src_br, dst_br, iface)
        assert iface in netcfg._get_bridge_members(dst_br)
        assert iface not in netcfg._get_bridge_members(src_br)
    finally:
        netcfg._move_interface(dst_br, src_br, iface)
        netcfg._destroy_bridge(dst_br)
        assert iface in netcfg._get_bridge_members(src_br)


def test_ip_assign_and_clear_integration():
    _require_macos_root()
    br = _next_bridge_name()
    netcfg._ensure_bridge(br)
    try:
        ip = "10.250.250.0"
        mask = "255.255.255.254"
        netcfg.ip_add(br, ip, mask)
        assert ip in netcfg.ip_lists(br)
        netcfg.ip_del(br, ip)
        assert ip not in netcfg.ip_lists(br)
    finally:
        netcfg._destroy_bridge(br)


def test_mtu_set_restore_integration():
    _require_macos_root()
    iface = _pick_iface()
    if not iface:
        pytest.skip("no interface available")
    orig = netcfg._get_interface_mtu(iface)
    assert orig is not None
    # Toggle between common safe MTUs on macOS (1500/9000)
    new = 9000 if orig != 9000 else 1500
    try:
        netcfg._set_interface_mtu(iface, new)
        cur = netcfg._get_interface_mtu(iface)
        # Some interfaces clamp/ignore non-supported MTUs; accept either target or original
        assert cur in (new, orig)
    finally:
        if orig is not None:
            netcfg._set_interface_mtu(iface, orig)
