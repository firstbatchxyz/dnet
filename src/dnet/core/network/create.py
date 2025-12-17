
import ipaddress
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Tuple
from dnet.core.types.topology import TopologyInfo
from dnet_p2p import DnetDeviceProperties
from dnet.utils.logger import logger
from dnet.core.types.topology import (
    TopologyInfo as TypesTopologyInfo,
) 

# Globals 
TB_MDP_COST = 10
ETH_MDP_COST = 20 
ETH_FDP_COST = 100
WIFI_FDB_COST = 200

MDP_IPV4_BASE = ipaddress.IPv4Address("10.101.0.0")
FDP_IPV4_BASE = ipaddress.IPv4Address("10.101.128.0")
AP_IPV4_BASE  = ipaddress.IPv4Address("10.101.160.0")

MDP_INC = 0
FDP_INV = 0
AP_INV = 0

# Models
# NOTE: for now the FDP route is empty
class NetworkRoute(BaseModel):
    node: str = Field(..., description="Name of neighbor node")
    mdp_self_ipv4: str = Field( ..., description="Primary path self IPv4")
    mdp_ipv4: str = Field( ..., description="Primary path IPv4")
    mdp_interface: str = Field( ..., description="inet for primary path")
    mdp_cost: int = Field(..., description="Routing cost for primary path")
    fdp_ipv4: str = Field(..., description="Fallback path IPv4")
    fdp_interface: str = Field( ..., description="inet for fallback path")
    fdp_cost: int = Field(..., description="Routing cost for fallback path")

class NodeNetworkPlan(BaseModel):
    instance: str = Field(..., description="")
    mgmt_ipv4: str = Field(..., description="Management IPv4")
    routes: List[NetworkRoute] = Field(
        default_factory=list, description="Per-neighbor route plans"
    )
    tb_bridges: List[str] = Field(
        default_factory=list,
        description="per‑port Thunderbolt bridge interface names",
    )

# L2/L3 topology for orchestration network.
class NetworkTopologyPlan(BaseModel):
    nodes: Dict[str, NodeNetworkPlan] = Field(
        default_factory=dict, description="Per‑node network plans"
    )

def _get_tb_link(a: DnetDeviceProperties, b: DnetDeviceProperties):
    if not a.thunderbolt and not b.thunderbolt:
        return (False, None, None)
    for b_host, b_connected in b.thunderbolt.instances:
        for b_con in b_connected:
            if hasattr(a.thunderbolt, "instances"):
                for a_host, _ in a.thunderbolt.instances:
                    if b_con.uuid == a_host.uuid:
                        return (True, a_host.uuid, b_host.uuid)
    return (False, None, None)

def topo_to_network(topo: TypesTopologyInfo) -> NetworkTopologyPlan:
    _reset_addr_inc()
    nodes: Dict[str, NodeNetworkPlan] = {}
    dev_by_name = {d.instance: d for d in topo.devices}
    br_count: Dict[str, int] = {}
    tb_uuid_to_bridge: Dict[str, Dict[str, str]] = {}
    links: set[tuple[str, str]] = set()

    for name, dev in dev_by_name.items(): 
        nodes[name] = NodeNetworkPlan(
            instance=name,
            mgmt_ipv4="", 
            routes=[],
            tb_bridges=[],
        )

    for ass in topo.assignments:
        if not ass.next_instance: continue
        link = tuple(sorted([ass.instance, ass.next_instance]))
        links.add(link)

    for a_name, b_name in links:
        a = dev_by_name.get(a_name)
        b = dev_by_name.get(b_name)
        connected, uuid_a, uuid_b = _get_tb_link(a, b)
        if not connected or not uuid_a or not uuid_b:
            # TODO: Fallback to ethernet or wifi
            continue

        # map bridges to connections
        if a_name not in tb_uuid_to_bridge: tb_uuid_to_bridge[a_name] = {}
        if uuid_a not in tb_uuid_to_bridge[a_name]:
            idx = br_count.get(a_name, 0) + 1
            br_count[a_name] = idx
            br_a = f"bridge{idx}"
            tb_uuid_to_bridge[a_name][uuid_a] = br_a
            if br_a not in nodes[a_name].tb_bridges:
                nodes[a_name].tb_bridges.append(br_a)
        else:
            br_a = tb_uuid_to_bridge[a_name][uuid_a]

        if b_name not in tb_uuid_to_bridge: tb_uuid_to_bridge[b_name] = {}
        if uuid_b not in tb_uuid_to_bridge[b_name]:
            idx = br_count.get(b_name, 0) + 1
            br_count[b_name] = idx
            br_b = f"bridge{idx}"
            tb_uuid_to_bridge[b_name][uuid_b] = br_b
            if br_b not in nodes[b_name].tb_bridges:
                nodes[b_name].tb_bridges.append(br_b)
        else:
            br_b = tb_uuid_to_bridge[b_name][uuid_b]

        ip_a, ip_b = _alloc_mdp_link() 
        nodes[a_name].routes.append(
            NetworkRoute(
                node=b_name,
                mdp_self_ipv4=ip_a,
                mdp_ipv4=ip_b,
                mdp_interface=br_a,
                mdp_cost=TB_MDP_COST,
                fdp_ipv4="",
                fdp_interface="",
                fdp_cost=0,
            )
        )
        nodes[b_name].routes.append(
            NetworkRoute(
                node=a_name,
                mdp_self_ipv4=ip_b,
                mdp_ipv4=ip_a,
                mdp_interface=br_b,
                mdp_cost=TB_MDP_COST,
                fdp_ipv4="",
                fdp_interface="",
                fdp_cost=0,
            )
        )
    return NetworkTopologyPlan(nodes=nodes)

def _reset_addr_inc():
    global MDP_INC
    global FDP_INC
    global AP_INC
    MDP_INC = 0
    FDP_INC = 0
    AP_INC = 0

def _alloc_mdp_link() -> Tuple[str, str]:
    global MDP_INC
    root = int(MDP_IPV4_BASE) + MDP_INC 
    MDP_INC += 2
    return f"{ipaddress.IPv4Address(root)}", f"{ipaddress.IPv4Address(root+1)}"

def _alloc_fdp_link() -> Tuple[str, str]:
    global FDP_INC
    root = int(FDP_IPV4_BASE) + FDP_INC 
    FDP_INC += 2
    return f"{ipaddress.IPv4Address(root)}", f"{ipaddress.IPv4Address(root+1)}"

def _alloc_ap_link() -> Tuple[str, str]:
    global AP_INC; 
    root = int(AP_IPV4_BASE) + AP_INC
    AP_INC += 2
    return f"{ipaddress.IPv4Address(root)}", f"{ipaddress.IPv4Address(root+1)}"
