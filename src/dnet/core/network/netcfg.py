#!/usr/bin/env python3

import os
import getpass
import socket
import re
import sys
import json
import argparse
import ipaddress
import subprocess
import socket
from typing import Optional


def is_perm_err(s: str) -> bool:
    return (
        "permission denied" in s
        or "operation not permitted" in s
        or "sioc" in s.lower()
    )


# Return: returncode, output, stderr
# NOTE: prints are written in stderr to separate from json output in stdout
def __exec(cmd: list[str]) -> tuple[int, str, str]:
    try:
        print("$ " + " ".join(cmd), file=sys.stderr)
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        if p.returncode != 0:
            print(
                f"WARN: Command failed ({p.returncode}): {' '.join(cmd)}\n{err}",
                file=sys.stderr,
            )
        return p.returncode, out, err
    except Exception as e:
        print(f"ERROR: Command execution failed for {cmd}: {e}", file=sys.stderr)
        return 127, "", str(e)


def _ensure_up(obj: str):
    rc, _, err = __exec(["ifconfig", obj, "up"])
    if rc != 0 and is_perm_err(err):
        raise PermissionError(err)


# Bridges and Interfaces =======


def _bridge_exists(name: str) -> bool:
    rc, _, _ = __exec(["ifconfig", name])
    return rc == 0


def _create_bridge(bridge: str) -> bool:
    if _bridge_exists(bridge):
        return True
    rc, _, err = __exec(["ifconfig", bridge, "create"])
    if rc != 0:
        if "File exists" not in err:
            raise RuntimeError(f"Failed to create {bridge}: {err}")
    return rc == 0


def _set_bridge_status(bridge: str, enabled: bool):
    op = "up" if enabled else "down"
    rc, _, err = __exec(["ifconfig", bridge, op])
    if rc != 0 and is_perm_err(err):
        raise PermissionError(err)
    return rc == 0


def _ensure_bridge(bridge: str) -> None:
    if not _bridge_exists(bridge):
        _create_bridge(bridge)
    _set_bridge_status(bridge, enabled=True)


def _get_bridge_members(name: str) -> list[str]:
    rc, out, _ = __exec(["ifconfig", name])
    if rc != 0:
        return []
    members: list[str] = []
    for line in out.splitlines():
        m = re.search(r"\bmember:\s+(\w+)\b", line)
        if m:
            members.append(m.group(1))
    return members


def _list_bridges() -> list[str]:
    rc, out, _ = __exec(["ifconfig", "-l"])
    if rc == 0 and out:
        ifs = out.split()
        return [i for i in ifs if i.startswith("bridge")]
    return []


def _destroy_bridge(br: Optional[str] = None) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    if br == "bridge0":
        return actions
    rc, _, err = __exec(["ifconfig", br, "destroy"])
    if rc != 0 and is_perm_err(err):
        raise PermissionError(err)
    actions.append({"destroyed": br})
    return actions


def _get_thunderbolt_interfaces() -> list[str]:
    rc, out, _ = __exec(["networksetup", "-listallhardwareports"])
    if rc != 0:
        return []
    cur_port = None
    en_list: list[str] = []
    for line in out.splitlines():
        if line.startswith("Hardware Port:"):
            cur_port = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and cur_port:
            dev = line.split(":", 1)[1].strip()
            if "Thunderbolt" in cur_port and dev.startswith("en"):
                en_list.append(dev)
            cur_port = None
    return en_list


def _get_tb_ports() -> list[dict[str, str]]:
    rc, out, err = __exec(["system_profiler", "SPThunderboltDataType", "-json"])
    if rc != 0:
        raise RuntimeError(err or "system_profiler_failed")
    try:
        data = json.loads(out or "{}")
    except Exception:
        data = {}
    arr = data.get("SPThunderboltDataType", []) or []
    ports: list[dict[str, str]] = []
    for item in arr:
        name = str(item.get("_name", ""))
        uuid = str(item.get("domain_uuid_key", ""))
        for k, v in item.items():
            if not isinstance(v, dict):
                continue
            if (
                not (k.startswith("receptacle_") and k.endswith("_tag"))
                and k != "receptacle_upstream_ambiguous_tag"
            ):
                continue
            status = str(v.get("receptacle_status_key", ""))
            if status and status in [
                "receptacle_connected",
                "receptacle_no_devices_connected",
            ]:
                rec = {
                    "bus": name,
                    "uuid": uuid,
                    "receptacle": str(v.get("receptacle_id_key", "")),
                    "status": status,
                    "speed": str(v.get("current_speed_key", "")),
                    "link": str(v.get("link_status_key", "")),
                }
                ports.append(rec)
    return ports


def _find_interface(en: str) -> Optional[str]:
    for br in _list_bridges():
        if en in _get_bridge_members(br):
            return br
    return None


# NOTE:
# Interfaces that are up can reject PROMISC mode transition
# forced by addm so we explicitly disable them before every move
def _move_interface(from_br: str, to_br: str, en: str) -> None:
    _ensure_bridge(to_br)
    _set_interface_status(en, enabled=False)
    _detach_interface(en, from_br)
    _attach_interface(en, to_br)
    _set_interface_status(en, enabled=True)
    _set_bridge_status(to_br, enabled=True)


def _set_interface_status(en: str, enabled: bool) -> list[dict[str, str]]:
    op = "up" if enabled else "down"
    rc, _, err = __exec(["ifconfig", en, op])
    if rc != 0 and is_perm_err(err):
        raise PermissionError(err)


def _set_interface_mtu(en: str, mtu: int) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    rc, _, err = __exec(["ifconfig", en, "mtu", mtu])
    if rc != 0 and is_perm_err(err):
        raise PermissionError(err)
    actions.append({"updated": en, "mtu": mtu})
    return actions

def _get_interface_mtu(en: str) -> Optional[int]:
    rc, out, _ = __exec(["ifconfig", en])
    if rc != 0 or not out:
        return None
    m = re.search(r"\bmtu\s+(\d+)\b", out)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _attach_interface(en: str, bridge: str) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    _set_interface_status(en, enabled=False)
    rc, _, err = __exec(["ifconfig", bridge, "addm", en])
    if rc != 0 and is_perm_err(err):
        raise PermissionError(err)
    _set_interface_status(en, enabled=True)
    actions.append({"attached": en, "to": bridge})
    return actions


def _detach_interface(en: str, br: str = None) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    if not br:
        return actions
    rc, _, err = __exec(["ifconfig", br, "deletem", en])
    if rc != 0 and is_perm_err(err):
        raise PermissionError(err)
    actions.append({"detached": en, "from": br})
    return actions


# IPv4 Control =======


def _assign_ip(ifname: str, ip: str, netmask: str) -> None:
    rc, _, err = __exec(["ifconfig", ifname, "inet", ip, netmask, "alias"])
    if rc != 0:
        if is_perm_err(err):
            raise PermissionError(err)
        rc2, _, err2 = __exec(["ifconfig", ifname, "inet", ip, netmask])
        if rc2 != 0 and is_perm_err(err2):
            raise PermissionError(err2)


def _route_to_peer(ifname: str, peer: str, netmask: str) -> int:
    rc, _, err = __exec(["route", "change", peer, "-interface", ifname])
    if rc != 0 and is_perm_err(err):
        raise PermissionError(err)
        return -1
    return 0


def ip_lists(ifname: str) -> list[str]:
    rc, out, _ = __exec(["ifconfig", ifname])
    addrs: list[str] = []
    if rc != 0:
        return addrs
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("inet ") and "inet6" not in line:
            parts = line.split()
            if len(parts) >= 2:
                addrs.append(parts[1])
    return addrs


def ip_add(ifname: str, ip: str, netmask: str) -> list[dict[str, str]]:
    _assign_ip(ifname, ip, netmask)
    _set_interface_status(ifname, enabled=True)
    return [{"ipv4": ip, "on": ifname, "netmask": netmask}]


def ip_del(ifname: str, ip: str) -> list[dict[str, str]]:
    rc, _, err = __exec(["ifconfig", ifname, "-alias", ip])
    if rc != 0 and is_perm_err(err):
        raise PermissionError(err)
    return [{"removed": ip, "on": ifname}]


def ip_clear(interfaces: Optional[list[str]] = None) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    ifs = interfaces or _list_bridges()
    for ifn in ifs:
        for ip in ip_lists(ifn):
            actions += ip_del(ifn, ip)
    return actions


#  Network =======


def apply_network_mappings(mappings: list[dict]) -> dict:
    bridges = {str(m.get("mdp_interface")) for m in mappings if m.get("mdp_interface")}
    for br in sorted(bridges):
        _ensure_bridge(br)

    assigned: dict[str, str] = {}
    used_ens: set[str] = set()
    current_b0 = set(_get_bridge_members("bridge0"))

    actions: list[dict[str, str]] = []

    for m in mappings:
        br_obj = m.get("mdp_interface")
        if not br_obj:
            continue
        target_br = str(br_obj)
        if not re.match(r"^(en|bridge)\d+$", target_br):
            raise ValueError("invalid_interface")
        if target_br in assigned:
            continue

        current_en = m.get("thunderbolt_en", m.get("tb_self_en"))
        if not current_en:
            print(
                f"WARN: No available Thunderbolt en* port to attach to {target_br}",
                file=sys.stderr,
            )
            continue
        if not re.match(r"^(en|bridge)\d+$", str(current_en)):
            raise ValueError("invalid_interface")
        if current_en in used_ens:
            raise RuntimeError("duplicate_interface")
        used_ens.add(current_en)

        from_br = _find_interface(str(current_en)) or ""
        _move_interface(from_br, target_br, current_en)
        assigned[target_br] = current_en
        actions.append(
            {"bridge": target_br, "member": current_en, "moved_from": from_br or "none"}
        )

    for m in mappings:
        self_ip = m.get("mdp_self_ipv4")
        peer_ip = m.get("mdp_ipv4")
        iface = m.get("mdp_interface")
        if self_ip and peer_ip and iface:
            try:
                ipaddress.IPv4Address(str(self_ip))
                ipaddress.IPv4Address(str(peer_ip))
            except Exception:
                raise ValueError("invalid_ip")
            netmask = "255.255.255.254"
            _assign_ip(iface, self_ip, netmask)
            if self_ip not in ip_lists(iface):
                raise RuntimeError("Failed assign IPv4 Address: {self_ip}")
            actions.append(
                {
                    "bridge": iface,
                    "addr": self_ip,
                    "netmask": netmask,
                }
            )
            if _route_to_peer(iface, peer_ip, netmask) < 0:
                raise RuntimeError(
                    "Failed to update routing table with peer IP: {peer_ip}"
                )
    return {
        "status": "ok",
        "actions": actions,
        "assigned": assigned,
        "current_bridge0": sorted(current_b0),
    }

# Sudoers registration ========

# visudo rejects unsafe paths and permissions
# add executable here for registration
def register_in_homebrew_bin() -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    dest = "/opt/homebrew/bin/dnet-netcfg"
    src = os.path.abspath(__file__)
    with open(src, "r", encoding="utf-8") as f:
        content = f.read()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(dest, 0o755)
    actions.append({"installed": dest, "from": src})
    user = os.environ.get("SUDO_USER") or os.environ.get("USER") or getpass.getuser()
    actions += register_sudoers(user)
    return actions


def register_sudoers(user: str) -> list[dict[str, str]]:
    bin_path = "/opt/homebrew/bin/dnet-netcfg"
    if not os.path.exists(bin_path):
        raise FileNotFoundError(bin_path)
    st = os.stat(bin_path)
    if (st.st_mode & 0o022) != 0:
        raise RuntimeError("unsafe_binary_permissions")
    line = f"{user} ALL=(root) NOPASSWD: {bin_path}\n"
    path = "/etc/sudoers.d/dnet-netcfg"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(line)
    os.chmod(tmp, 0o440)
    rc, _, err = __exec(["visudo", "-cf", tmp])
    if rc != 0:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise RuntimeError("sudoers_validation_failed")
    os.replace(tmp, path)
    return [{"wrote": path, "user": user, "bin": bin_path}]


def remove_sudoers() -> list[dict[str, str]]:
    path = "/etc/sudoers.d/dnet-netcfg"
    try:
        os.remove(path)
        rc = 0
    except FileNotFoundError:
        rc = 0
    except PermissionError as e:
        raise PermissionError(str(e))
    return [{"removed": path, "rc": str(rc)}]


def reset_network(also_remove_sudoers: bool = False) -> dict:
    actions: list[dict[str, str]] = []
    dest = "bridge0"
    _ensure_bridge(dest)
    _ensure_up(dest)
    for br in _list_bridges():
        if br == dest: continue
        for en in _get_bridge_members(br):
            _move_interface(br, dest, en)
            actions.append({"moved": en, "from": br, "to": dest})
    for br in _list_bridges():
        if br == dest:
            continue
        actions += _destroy_bridge(br)
    if also_remove_sudoers:
        actions += remove_sudoers()
    return {"status": "ok", "reset": True, "actions": actions}


def status() -> dict:
    brs = _list_bridges()
    members = {br: _get_bridge_members(br) for br in brs}
    tb = _get_thunderbolt_interfaces()
    addrs = {br: ip_lists(br) for br in brs}
    return {
        "ok": True,
        "bridges": brs,
        "members": members,
        "thunderbolt": tb,
        "addresses": addrs,
    }


# Network plan execution ======

def read_plan(path: Optional[str]) -> list[dict]:
    if path:
        with open(path, "r") as f:
            obj = json.load(f)
    else:
        data = sys.stdin.read()
        obj = json.loads(data)
    if isinstance(obj, dict) and "mappings" in obj:
        return list(obj["mappings"])
    if isinstance(obj, list):
        return obj
    raise SystemExit(2)


def _validate_plan(mappings: list[dict]) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    used_en: set[str] = set()
    for i, m in enumerate(mappings):
        iface = m.get("mdp_interface")
        if not iface or not re.match(r"^(en|bridge)\d+$", str(iface)):
            errors.append(f"mapping[{i}]: invalid_interface")
        en = m.get("thunderbolt_en") or m.get("tb_self_en")
        if en and not re.match(r"^en\d+$", str(en)):
            errors.append(f"mapping[{i}]: invalid_member")
        if en:
            if str(en) in used_en:
                errors.append(f"mapping[{i}]: duplicate_interface:{en}")
            used_en.add(str(en))
        sip = m.get("mdp_self_ipv4")
        pip = m.get("mdp_ipv4")
        try:
            ipaddress.IPv4Address(str(sip))
            ipaddress.IPv4Address(str(pip))
        except Exception:
            errors.append(f"mapping[{i}]: invalid_ip")
    ok = len(errors) == 0
    return {"ok": ok, "errors": errors, "warnings": warnings, "count": len(mappings)}


def _test_ping(peer: str, count: int = 2) -> dict:
    rc, out, err = __exec(["ping", "-n", "-c", str(int(count)), peer])
    return {"ok": rc == 0, "peer": peer, "exit": rc, "stdout": out}


def _test_route(peer: str, expect: Optional[str] = None) -> dict:
    rc, out, err = __exec(["route", "-n", "get", peer])
    iface = dest = gw = None
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("interface:"):
            iface = line.split(":", 1)[1].strip()
        elif line.startswith("destination:"):
            dest = line.split(":", 1)[1].strip()
        elif line.startswith("gateway:"):
            gw = line.split(":", 1)[1].strip()
    direct = bool(dest and dest != "default" and gw and gw.startswith("link#"))
    ok = bool(rc == 0 and iface and (expect is None or iface == expect))
    return {
        "ok": ok,
        "peer": peer,
        "interface": iface or "",
        "expect": expect or "",
        "destination": dest or "",
        "gateway": gw or "",
        "direct": direct,
        "exit": rc,
    }


# ping and iperf3 ======

def _test_listen(port: int, token: Optional[str] = None, timeout: float = 5.0) -> dict:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(float(timeout))
    s.bind(("0.0.0.0", int(port)))
    try:
        data, addr = s.recvfrom(2048)
        ok = True
        text = data.decode(errors="ignore")
        if token:
            ok = text.strip() == str(token)
        try:
            s.sendto(data, addr)
        except Exception:
            pass
        return {"ok": ok, "from": f"{addr[0]}:{addr[1]}", "data": text}
    except socket.timeout:
        return {"ok": False, "error": "timeout"}
    finally:
        try:
            s.close()
        except Exception:
            pass


def _test_iperf3_server(port: int = 5201, bind: Optional[str] = None, one_shot: bool = True) -> dict:
    cmd = ["iperf3", "-s"]
    if one_shot:
        cmd.append("-1")
    cmd += ["-p", str(int(port))]
    if bind:
        cmd += ["-B", str(bind)]
    rc, out, err = __exec(cmd)
    return {"ok": rc == 0, "port": port, "bind": bind or "", "exit": rc, "stdout": out}


def _test_iperf3_client(peer: str, port: int = 5201, time: int = 3) -> dict:
    cmd = ["iperf3", "-c", str(peer), "-p", str(int(port)), "-t", str(int(time)), "-J"]
    rc, out, err = __exec(cmd)
    res: dict = {"ok": rc == 0, "peer": peer, "port": port, "exit": rc}
    if out:
        try:
            data = json.loads(out)
            end = data.get("end", {}) if isinstance(data, dict) else {}
            sum_r = end.get("sum_received", {}) if isinstance(end, dict) else {}
            sum_s = end.get("sum_sent", {}) if isinstance(end, dict) else {}
            bps = sum_r.get("bits_per_second") or sum_s.get("bits_per_second")
            if bps is not None:
                res["bits_per_second"] = bps
            res["iperf3"] = data
        except Exception:
            res["stdout"] = out
    return res


# persistent service ==== 

def _service_exists(name: str) -> bool:
    rc, out, _ = __exec(["networksetup", "-listallnetworkservices"])
    if rc != 0:
        return False
    for line in (out or "").splitlines():
        line = line.strip()
        if not line or line.startswith("An asterisk"):
            continue
        if line.startswith("*"):
            line = line[1:].strip()
        if line == name:
            return True
    return False


def _ensure_service_for_iface(service: str, iface: str) -> None:
    if not _service_exists(service):
        __exec(["networksetup", "-createnetworkservice", service, iface])


def _persist_mappings(mappings: list[dict]) -> dict:
    actions: list[dict[str, str]] = []
    for m in mappings:
        iface = m.get("mdp_interface")
        self_ip = m.get("mdp_self_ipv4")
        if not iface or not self_ip:
            continue
        iface = str(iface)
        self_ip = str(self_ip)
        service = f"DNET-{iface}"
        _ensure_service_for_iface(service, iface)
        netmask = "255.255.255.254"
        __exec(["ipconfig", "set", iface, "MANUAL", self_ip, netmask])
        actions.append({"persisted": iface, "service": service, "ip": self_ip, "netmask": netmask})
    return {"ok": True, "actions": actions}


# argument parsing ====

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="dnet-netcfg",
        description="Utility to control bridges and Thunderbolt interfaces on macOS",
    )
    ap.add_argument(
        "--file",
        "-f",
        help="Path to JSON plan for apply mode. If omitted, reads stdin.",
    )

    sub = ap.add_subparsers(dest="cmd")

    # APPLY
    p_apply = sub.add_parser("apply", help="Apply network mappings from JSON plan")
    p_apply.add_argument("--file", "-f", help="Path to JSON plan (else read stdin)")

    # STATUS
    sub.add_parser(
        "status", help="Show bridges, members, Thunderbolt interfaces, and addresses"
    )

    # NETWORK
    p_net = sub.add_parser("network", help="Network operations")
    net = p_net.add_subparsers(dest="net_cmd")

    p_net_apply = net.add_parser("apply", help="Apply network mappings from JSON plan")
    p_net_apply.add_argument("--file", "-f", help="Path to JSON plan (else read stdin)")

    p_net_validate = net.add_parser("validate", help="Validate plan JSON")
    p_net_validate.add_argument("--file", "-f", help="Path to JSON plan (else read stdin)")

    p_net_reset = net.add_parser(
        "reset", help="Reset members to bridge0 and destroy extras"
    )
    p_net_reset.add_argument(
        "--also-remove-sudoers",
        action="store_true",
        help="Also remove sudoers rule during reset",
    )

    net.add_parser(
        "status", help="Show bridges, members, Thunderbolt interfaces, and addresses"
    )

    # TEST COMMANDS (minimal)
    p_test = sub.add_parser("test", help="Connectivity tests")
    test = p_test.add_subparsers(dest="test_cmd")
    p_t_ping = test.add_parser("ping", help="ICMP ping a peer")
    p_t_ping.add_argument("peer", help="Peer IPv4")
    p_t_ping.add_argument("--count", type=int, default=2)
    p_t_route = test.add_parser("route", help="Show route to a peer")
    p_t_route.add_argument("peer", help="Peer IPv4")
    p_t_route.add_argument("--expect", help="Expected egress interface (e.g. bridge1)")
    p_t_is = test.add_parser("iperf3-server", help="Run iperf3 in server mode (one-shot)")
    p_t_is.add_argument("--port", type=int, default=5201)
    p_t_is.add_argument("--bind", default=None, help="Bind IP")
    p_t_ic = test.add_parser("iperf3-client", help="Run iperf3 client to peer")
    p_t_ic.add_argument("peer", help="Peer IPv4")
    p_t_ic.add_argument("--port", type=int, default=5201)
    p_t_ic.add_argument("--time", type=int, default=3)

    # BRIDGE COMMANDS
    p_br = sub.add_parser("bridge", help="Bridge operations")
    br = p_br.add_subparsers(dest="br_cmd")
    p_br_create = br.add_parser("create", help="Create and bring up bridges")
    p_br_create.add_argument("name", nargs="+", help="Bridge name(s)")
    p_br_destroy = br.add_parser("destroy", help="Destroy bridge")
    p_br_destroy.add_argument("bridge", help="Specific bridge to destroy")

    p_br_attach = br.add_parser("attach", help="Attach interface to bridge")
    p_br_attach.add_argument("iface", help="Interface (e.g. en3)")
    p_br_attach.add_argument("bridge", help="Bridge (e.g. bridge1)")

    p_br_detach = br.add_parser("detach", help="Detach interface from bridge")
    p_br_detach.add_argument("iface", help="Interface (e.g., en3)")
    p_br_detach.add_argument("bridge", help="Bridge (e.g. bridge0)")

    p_br_move = br.add_parser(
        "move", help="Move member interface from source bridges to destination"
    )
    p_br_move.add_argument("iface", help="Interface (e.g. en3)")
    p_br_move.add_argument("src_br", help="Source bridge")
    p_br_move.add_argument("dst_br", default="bridge0", help="Destination bridge")

    # IFACE COMMANDS
    p_if = sub.add_parser("iface", help="Interface operations")
    iff = p_if.add_subparsers(dest="iface_cmd")

    p_if_add = iff.add_parser("ip-add", help="Add IPv4 to interface")
    p_if_add.add_argument("iface", help="Interface")
    p_if_add.add_argument("ip", help="IPv4")
    p_if_add.add_argument("netmask", help="Netmask")

    p_if_del = iff.add_parser("ip-del", help="Remove IPv4 from interface")
    p_if_del.add_argument("iface", help="Interface")
    p_if_del.add_argument("ip", help="IPv4")

    p_if_mtu = iff.add_parser("mtu", help="Set MTU value for interface.")
    p_if_mtu.add_argument("iface", help="Interface")
    p_if_mtu.add_argument("rate", help="MTU Value")

    # THUNDERBOLT COMMANDS
    p_tb = sub.add_parser("thunderbolt", help="Thunderbolt operations")
    tb = p_tb.add_subparsers(dest="tb_cmd")
    tb.add_parser("list", help="List Thunderbolt interfaces")
    p_tb_rec = tb.add_parser("ports", help="List Thunderbolt ports with status filtering")
    p_tb_rec.add_argument(
        "--status",
        action="append",
        dest="status",
        help="Filter by status (repeatable). Defaults to connected/no_devices_connected",
    )

    p_sudo = sub.add_parser("sudoers", help="Sudoers operations")
    sd = p_sudo.add_subparsers(dest="sudo_cmd")
    p_sudo_reg = sd.add_parser("register", help="Register sudoers rule for dnet-netcfg")
    p_sudo_reg.add_argument("--user", required=True, help="Username to grant NOPASSWD")
    sd.add_parser("remove", help="Remove sudoers rule for dnet-netcfg")

    p_reset = sub.add_parser(
        "reset", help="Reset members back and destroy extra bridges"
    )
    p_reset.add_argument(
        "--also-remove-sudoers",
        action="store_true",
        help="Also remove sudoers rule during reset",
    )

    ns = ap.parse_args()
    try:
        cmd = ns.cmd

        # Read config file and execute
        if not cmd:
            if not ns.file and sys.stdin.isatty():
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "error": "usage",
                            "message": "Use a subcommand or provide --file/STDIN for apply.",
                        }
                    )
                )
                sys.exit(2)
            plan = read_plan(ns.file)
            res = apply_network_mappings(plan)
            ok = bool(isinstance(res, dict) and res.get("actions"))
            out = {"ok": ok}
            if isinstance(res, dict):
                out.update(res)
            print(json.dumps(out))
            sys.exit(0 if ok else 3)

        if cmd == "apply":
            plan = read_plan(ns.file)
            res = apply_network_mappings(plan)
            ok = bool(isinstance(res, dict) and res.get("actions"))
            out = {"ok": ok}
            if isinstance(res, dict):
                out.update(res)
            print(json.dumps(out))
            sys.exit(0 if ok else 3)

        if cmd == "status":
            print(json.dumps(status()))
            sys.exit(0)

        if cmd == "reset":
            r = reset_network(
                also_remove_sudoers=bool(getattr(ns, "also_remove_sudoers", False))
            )
            actions = r.get("actions", [])
            print(json.dumps({"ok": True, "actions": actions}))
            sys.exit(0)

        if cmd == "network":
            if not ns.net_cmd:
                p_net.print_help()
                sys.exit(2)

            match ns.net_cmd:
                case "apply":
                    plan = read_plan(ns.file)
                    res = apply_network_mappings(plan)
                    ok = bool(isinstance(res, dict) and res.get("actions"))
                    out = {"ok": ok}
                    if isinstance(res, dict):
                        out.update(res)
                    actions = out
                case "validate":
                    plan = read_plan(ns.file)
                    out = _validate_plan(plan)
                    print(json.dumps(out))
                    sys.exit(0 if out.get("ok") else 3)
                case "reset":
                    r = reset_network(
                        also_remove_sudoers=bool(
                            getattr(ns, "also_remove_sudoers", False)
                        )
                    )
                    actions = r.get("actions", [])
                case "status":
                    print(json.dumps(status()))
                    sys.exit(0)

            print(json.dumps({"ok": True, "actions": actions}))
            sys.exit(0)

        if cmd == "bridge":
            if not ns.br_cmd:
                p_br.print_help()
                sys.exit(2)

            match ns.br_cmd:
                case "create":
                    actions: list[dict[str, str]] = []
                    for name in ns.name:
                        _ensure_bridge(name)
                        actions.append({"ensured": name})
                case "destroy":
                    actions = _destroy_bridge(ns.bridge)
                case "attach":
                    actions = _attach_interface(ns.iface, ns.bridge)
                case "detach":
                    actions = _detach_interface(ns.iface, from_bridge=ns.bridge)
                case "move":
                    iface = ns.iface
                    source = ns.src_br
                    dest = ns.dst_br or "bridge0"
                    _ensure_bridge(ns.dst_br)
                    actions: list[dict[str, str]] = []
                    if ns.src_br != ns.dst_br and iface in _get_bridge_members(
                        ns.src_br
                    ):
                        _move_interface(ns.src_br, ns.dst_br, ns.iface)
                        actions.append(
                            {"moved": ns.iface, "from": ns.src_br, "to": ns.dst_br}
                        )

            print(json.dumps({"ok": True, "actions": actions}))
            sys.exit(0)

        if cmd == "iface":
            if not ns.iface_cmd:
                p_if.print_help()
                sys.exit(2)

            match ns.iface_cmd:
                case "ip-add":
                    actions = ip_add(ns.iface, ns.ip, ns.netmask)
                case "ip-del":
                    actions = ip_del(ns.iface, ns.ip)
                case "mtu":
                    actions = _set_interface_mtu(ns.iface, ns.rate)

            print(json.dumps({"ok": True, "actions": actions}))
            sys.exit(0)

        if cmd == "test":
            if not ns.test_cmd:
                p_test.print_help()
                sys.exit(2)
            if ns.test_cmd == "ping":
                out = _test_ping(ns.peer, int(getattr(ns, "count", 2)))
                print(json.dumps(out))
                sys.exit(0 if out.get("ok") else 1)
            if ns.test_cmd == "route":
                out = _test_route(ns.peer, getattr(ns, "expect", None))
                print(json.dumps(out))
                sys.exit(0 if out.get("ok") else 1)
            if ns.test_cmd == "iperf3-server":
                out = _test_iperf3_server(int(getattr(ns, "port", 5201)), getattr(ns, "bind", None))
                print(json.dumps(out))
                sys.exit(0 if out.get("ok") else 1)
            if ns.test_cmd == "iperf3-client":
                out = _test_iperf3_client(ns.peer, int(getattr(ns, "port", 5201)), int(getattr(ns, "time", 3)))
                print(json.dumps(out))
                sys.exit(0 if out.get("ok") else 1)

        if cmd == "thunderbolt":
            if not ns.tb_cmd:
                p_tb.print_help()
                sys.exit(2)
            if ns.tb_cmd == "list":
                print(
                    json.dumps(
                        {"ok": True, "thunderbolt": _get_thunderbolt_interfaces()}
                    )
                )
                sys.exit(0)
            if ns.tb_cmd == "ports":
                print(json.dumps({"ok": True, "ports": _get_tb_ports()}))
                sys.exit(0)

        if cmd == "sudoers":
            if not ns.sudo_cmd:
                p_sudo.print_help()
                sys.exit(2)
            if ns.sudo_cmd == "register":
                actions = register_in_homebrew_bin()
                actions += register_sudoers(ns.user)
                print(json.dumps({"ok": True, "actions": actions}))
                sys.exit(0)
            if ns.sudo_cmd == "remove":
                actions = remove_sudoers()
                print(json.dumps({"ok": True, "actions": actions}))
                sys.exit(0)

        print(
            json.dumps({"ok": False, "error": "usage", "message": "Unknown subcommand"})
        )
        sys.exit(2)
    except PermissionError as e:
        print(json.dumps({"ok": False, "error": "permission", "message": str(e)}))
        sys.exit(13)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "exception", "message": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
