"""API utilities for ring topology generation."""

import asyncio
from typing import AsyncGenerator, Dict, Tuple
from dnet_p2p import DnetDeviceProperties, ThunderboltConnection
import mlx.core as mx
import numpy as np
from distilp import DeviceProfile
from distilp.components.dense_common import HALDAResult

from dnet.protos import dnet_ring_pb2
from dnet.protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from dnet.utils.logger import logger
from dnet.utils.time import utc_epoch_now
from dnet.ring.common import LayerAssignment


from .models import ChatParams


def create_generate_step_for_ring_with_grpc(
    stub: DnetRingServiceStub,
    *,
    callback_protocol: str = "http",
    callback_addr: str | None = None,
    compression: float = 0.0,
):
    pb2 = dnet_ring_pb2

    async def generate_step(
        nonce: str,
        node_origin: str,
        prompt: mx.array,
        pending_requests: Dict[str, asyncio.Future],
        params: ChatParams,
    ) -> AsyncGenerator[Tuple[mx.array, mx.array], None]:
        repetition_penalty = params.repetition_penalty
        repetition_context_size = params.repetition_context_size or 20

        reset_response = await stub.ResetCache(pb2.ResetCacheRequest())  # type: ignore
        logger.info("ResetCache Response: %s", reset_response.message)

        y = prompt
        repetition_context = prompt.tolist()
        if repetition_context_size:
            repetition_context = repetition_context[-repetition_context_size:]  # type: ignore

        async def _step(y):
            nonlocal repetition_context

            future = asyncio.get_running_loop().create_future()
            # Record API-side monotonic start time for end-to-end latency
            try:
                future._api_t0 = asyncio.get_running_loop().time()  # type: ignore[attr-defined]
            except Exception:
                pass
            pending_requests[nonce] = future

            # send the activation to next shards
            # Build callback target
            if callback_protocol == "grpc" and callback_addr:
                callback_url = f"grpc://{callback_addr}"
            else:
                callback_url = f"http://{node_origin}/results/{nonce}"

            # Build token payload
            tok_np = np.array(y, dtype=np.int32)
            act = pb2.Activation(
                batch_size=1,
                shape=[int(tok_np.size)],
                dtype="tokens",
                layer_id=-1,
                data=tok_np.tobytes(order="C"),
            )

            activation_message = pb2.ActivationRequest(
                nonce=nonce,
                node_origin=node_origin,
                timestamp=utc_epoch_now(),
                activation=act,
                callback_url=callback_url,
            )
            t_rpc = asyncio.get_running_loop().time()
            response = await stub.SendActivation(activation_message)  # type: ignore
            api_rpc_ms = (asyncio.get_running_loop().time() - t_rpc) * 1000.0
            logger.info(
                f"[PROFILE][API-TX] nonce={nonce} to=shard0 rpc_ms={api_rpc_ms:.2f} payload_kb={(len(activation_message.activation.data) / 1024):.1f}"
            )
            if not response.success:
                raise RuntimeError(
                    f"Sending activation {nonce} to {node_origin} was not succesful"
                )

            # Wait for token callback from last shard
            try:
                result = await asyncio.wait_for(future, timeout=3000.0)
            except asyncio.TimeoutError as e:
                raise RuntimeError(
                    "Did not receive token corresponding to %s, err: %s", nonce, e
                )
            except Exception as e:
                raise RuntimeError(f"Request {nonce} failed with exception {e!r}")
            finally:
                pending_requests.pop(nonce, None)

            # Only token IDs are supported (end shard samples inline)
            if not isinstance(result, int):
                raise RuntimeError(
                    "Expected token callback from shard, but received activation payload"
                )
            y = mx.array([int(result)], dtype=mx.int32)
            logprobs = mx.array([], dtype=mx.float32)
            if repetition_penalty:
                repetition_context.append(y.item())  # type: ignore # FIXME: !!!
            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:  # type: ignore # FIXME: !!!
                    repetition_context = repetition_context[-repetition_context_size:]  # type: ignore # FIXME: !!!
            return y, logprobs

        y, logprobs = await _step(y)
        mx.async_eval(y)
        while True:
            next_y, next_logprobs = await _step(y)
            mx.async_eval(next_y)
            yield y.item(), logprobs  # type: ignore # FIXME: !!!
            y, logprobs = next_y, next_logprobs

    return generate_step


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
    shards: Dict[str, DnetDeviceProperties],
    solution_w: list[int],
    solution_n: list[int],
    solution_k: int,
) -> list[LayerAssignment]:
    """Compute round-aware layer assignments, next node mapping, and prefetch windows from solver output.

    Args:
        shard_names: Name of the shards in the order they were given to the solver.
        shards: Discovered shards (some may be extra & unused)
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
        window_sizes[instance] = max(1, solution_w[i] // solution_k)

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

        # Add best candidate (or any remaining if no TB connections exist)
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
