from .base import make_policy, ComputePolicy
from . import fit_in_memory, offload
from .noop import NoopPolicy
from .fit_in_memory import FitInMemoryPolicy
from .offload import OffloadPolicy
from dnet.config import TopologySettings
from dataclasses import dataclass
from typing import Type, Union


@dataclass
class PolicyPlan:
    mode: str
    window_size: int
    resident_windows: int
    is_sliding: bool
    policy_cls: Type[ComputePolicy]


def plan_policy(
    *,
    local_count: int,
    requested_w: int,
    residency_size: int,
    topology_config: Union[TopologySettings, "TopologySettings"],
) -> PolicyPlan:
    requested_w = max(1, requested_w)
    n_residency = max(1, residency_size)

    if n_residency < requested_w:
        # sliding-fit flavor
        mode = "offload"  # external name
        sliding = True
        resident_windows = 1
        window_size = max(1, min(n_residency, local_count))
    else:
        if requested_w >= local_count:
            mode = "fit"
            sliding = False
            resident_windows = (
                9999  # TODO: not sure of how resident windows are assigned
            )
            window_size = local_count
        else:
            mode = "offload"
            sliding = False
            resident_windows = topology_config.resident_windows
            window_size = max(1, min(requested_w, local_count))

    if mode == "fit":
        return PolicyPlan(
            mode=mode,
            window_size=window_size,
            resident_windows=resident_windows,
            policy_cls=FitInMemoryPolicy,
            is_sliding=sliding,
        )
    else:
        return PolicyPlan(
            mode=mode,
            window_size=window_size,
            resident_windows=resident_windows,
            policy_cls=OffloadPolicy,
            is_sliding=sliding,
        )


__all__ = [
    "make_policy",
    "ComputePolicy",
    "plan_policy",
    "PolicyPlan",
    "fit_in_memory",
    "offload",
    "NoopPolicy",
]
