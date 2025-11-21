"""API utilities for ring topology generation."""

from typing import Dict
from dnet_p2p import ThunderboltConnection
import numpy as np
from distilp.common import DeviceProfile
from distilp.solver import HALDAResult
from dnet.utils.logger import logger
from dnet.core.types.topology import LayerAssignment


def postprocess_single_round(
    device_names: list[str], solution: HALDAResult
) -> tuple[list[str], HALDAResult]:
    """Postprocess HALDA solution for single-round case (k=1).

    This will adjust the assignments so that single-layer assigned devices are ignored, and
    those layers are instead given to their immediate neighbors."""
    if solution.k != 1:
        return (device_names, solution)  # do nothing for k > 1

    # keep doing this until no single-layer devices remain
    # FIXME: do this more performant
    while 1 in solution.w:
        for i, w in enumerate(solution.w):
            if w == 1:
                # NOTE: we are not checking `n` here in particular but assuming its `<= w`,
                # we are safe to increment that on the target device

                # assigned a single layer, reassign to neighbor (wrapping around)
                left_idx = (i - 1) % len(solution.w)
                right_idx = (i + 1) % len(solution.w)

                # prefer neighbor with fewer assigned layers
                if solution.w[left_idx] < solution.w[right_idx]:
                    chosen_idx = left_idx
                elif solution.w[right_idx] < solution.w[left_idx]:
                    chosen_idx = right_idx
                else:  # equal, 50-50 chance
                    if np.random.rand() < 0.5:
                        chosen_idx = left_idx
                    else:
                        chosen_idx = right_idx

                # update solution values
                solution.w[chosen_idx] += 1
                solution.n[chosen_idx] += 1

                # remove device from solution & names
                solution.w.pop(i)
                solution.n.pop(i)
                popped = device_names.pop(i)
                logger.info(
                    f"Removed device {popped} with single layer, reassigned to neighbor"
                )

                break  # continue with outer while loop

    return (device_names, solution)


def compute_layer_assignments(
    shard_order: list[str],
    solution_w: list[int],
    solution_n: list[int],
    solution_k: int,
) -> list[LayerAssignment]:
    """Compute round-aware layer assignments, next node mapping, and prefetch windows from solver output.

    Args:
        shard_order: Shards in order around the ring.
        solution_w: Solver result `w` for list of assigned layers.
        solution_n: Solver result `n` for number of GPU resident layers.
        solution_k: Solver result `k` for number of rounds.

    Returns:
        Layer assignments for each shard, in the same order as `shard_names`.
    """
    if len(solution_w) != len(shard_order) or len(solution_n) != len(shard_order):
        raise ValueError(
            f"Device count mismatch: w={len(solution_w)}, n={len(solution_n)}, devices={len(shard_order)}"
        )

    # number of layers equal to total assigned layers across all devices over all rounds
    num_layers = sum(solution_w) * solution_k
    logger.info(
        f"Distributing {num_layers} layers to {len(shard_order)} devices in {solution_k} rounds",
    )

    # compute assigned layers
    current_layer = 0
    assigned_layers: Dict[str, list[list[int]]] = {
        name: [[] for _ in range(solution_k)] for name in shard_order
    }
    for round_idx in range(solution_k):
        for device_idx, device_name in enumerate(shard_order):
            for _ in range(solution_w[device_idx]):
                assigned_layers[device_name][round_idx].append(current_layer)
                current_layer += 1

    assert current_layer == num_layers, (
        f"Assigned {current_layer} layers, expected {num_layers}"
    )

    # compute residency & windows sizes & neighbors for each shard
    residency_sizes: Dict[str, int] = {}
    window_sizes: Dict[str, int] = {}
    next_instances: Dict[str, str] = {}
    for i, instance in enumerate(shard_order):
        residency_sizes[instance] = solution_n[i]
        window_sizes[instance] = solution_w[i]

        # `dev_1 -> dev_2 -> ... -> dev_n -> dev_1` (wraps around)
        if i < len(shard_order) - 1:
            next_instances[instance] = shard_order[i + 1]
        else:
            next_instances[instance] = shard_order[0]  # wrap around

    assignments = [
        LayerAssignment(
            instance=name,
            layers=assigned_layers[name],
            next_instance=next_instances[name],
            window_size=window_sizes[name],
            residency_size=residency_sizes[name],
        )
        for name in shard_order
    ]

    logger.info("Assignments: %s", assignments)
    return assignments


def optimize_device_ordering(
    shard_profiles: Dict[str, DeviceProfile],
    thunderbolt_conns: Dict[str, Dict[str, ThunderboltConnection]],
) -> list[str]:
    """Optimize device ordering to place Thunderbolt-connected devices adjacently.

    Args:
        shard_profiles: Collected shard profiles
        thunderbolt_conns: Thunderbolt connections mapping (device -> {neighbor -> connection_info})

    Returns:
        Optimized list of device names
    """
    device_names = list(shard_profiles.keys())

    # FIXME: shards on the same machine should be adjacent too!

    # Build adjacency graph of Thunderbolt connections
    # Graph: device_name -> set of connected device names
    tb_graph: Dict[str, set[str]] = {name: set() for name in device_names}
    for device_name, neighbors in thunderbolt_conns.items():
        if device_name in tb_graph:
            for neighbor_name in neighbors.keys():
                if neighbor_name in tb_graph:
                    tb_graph[device_name].add(neighbor_name)
                    tb_graph[neighbor_name].add(device_name)

    # Greedy ordering: Start with all head devices, then pick neighbors with most TB connections
    ordered: list[str] = []
    remaining: set[str] = set(device_names)

    while remaining:
        best_candidate = None
        best_score = -1

        # For each remaining device, calculate connection score to already-ordered devices
        for candidate in remaining:
            # Count Thunderbolt connections to devices already in the order
            score = sum(
                1 for ordered_dev in ordered if ordered_dev in tb_graph[candidate]
            )

            # Prioritize devices with TB connections, otherwise any device is fine
            if score > best_score:
                best_score = score
                best_candidate = candidate

        # Add the best candidate (or any remaining if no TB connections exist)
        if best_candidate:
            ordered.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            # Fallback: just pick any remaining device
            next_device = remaining.pop()
            ordered.append(next_device)

    logger.info("Optimized device ordering: %s", ordered)
    logger.info("Thunderbolt graph: %s", tb_graph)

    return ordered
