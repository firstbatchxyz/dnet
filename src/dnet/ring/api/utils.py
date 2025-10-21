"""API utilities for ring topology generation."""

import asyncio
from typing import AsyncGenerator, Dict, Tuple, Optional
from dnet_p2p import DnetDeviceProperties
from dnet_p2p.thunderbolt import ThunderboltConnection
import mlx.core as mx
import numpy as np
from distilp import DeviceProfile

from dnet.protos import dnet_ring_pb2
from dnet.protos.dnet_ring_pb2_grpc import DnetRingServiceStub
from dnet.utils.logger import logger
from dnet.utils.time import utc_epoch_now
from dnet.ring.common import LayerAssignment


from .models import ChatBaseParams


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
        params: ChatBaseParams,
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
                repetition_context.append(y.item())
            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:
                    repetition_context = repetition_context[-repetition_context_size:]
            return y, logprobs

        y, logprobs = await _step(y)
        mx.async_eval(y)
        while True:
            next_y, next_logprobs = await _step(y)
            mx.async_eval(next_y)
            yield y.item(), logprobs
            y, logprobs = next_y, next_logprobs

    return generate_step


def compute_layer_assignments(
    device_names: list[str],
    solution_w: list[int],
    solution_k: int,
    shards: Dict[str, DnetDeviceProperties],
) -> list[LayerAssignment]:
    """Compute round-aware layer assignments, next node mapping, and prefetch windows from solver output.

    Args:
        device_names: Device names in solver order
        solution_w: Solver result `w` for list of assigned layers.
        solution_k: Solver result `k` for number of rounds.
        shards: Discovered shards

    Returns:
        Tuple of (layer assignments per device per round, next service per device in ring, prefetch window per device)
    """
    if len(solution_w) != len(shards) or len(device_names) != len(shards):
        raise ValueError(
            f"Device count mismatch: solution={len(solution_w)}, shards={len(shards)}"
        )

    num_layers = sum(solution_w) * solution_k
    logger.info(
        "Distributing %d layers to %d devices in %d rounds",
        num_layers,
        len(shards),
        solution_k,
    )

    layer_assignments: Dict[str, list[list[int]]] = {
        name: [[] for _ in range(solution_k)] for name in device_names
    }
    current_layer = 0
    for round_idx in range(solution_k):
        for device_idx, device_name in enumerate(device_names):
            for _ in range(solution_w[device_idx]):
                layer_assignments[device_name][round_idx].append(current_layer)
                current_layer += 1
    assert current_layer == num_layers, (
        f"Assigned {current_layer} layers, expected {num_layers}"
    )

    # Compute next service for each device in ring topology
    # In ring: dev1 -> dev2 -> ... -> devN -> dev1 (wraps around)
    # Each shard will detect when processing the final layer and send to API
    next_service_map: Dict[str, Optional[str]] = {}

    if len(device_names) == 1:
        # Single device: forwards to itself in a loop
        next_service_map[device_names[0]] = device_names[0]
        logger.info("Ring (single device): %s -> SELF (loops back)", device_names[0])
    else:
        # Multiple devices: each forwards to the next in the ring
        for i, service_name in enumerate(device_names):
            if i < len(device_names) - 1:
                # Forward to next device
                next_service_map[service_name] = device_names[i + 1]
            else:
                # Last device wraps to first device
                next_service_map[service_name] = device_names[0]

        # Log ring topology
        for service_name in device_names:
            logger.info("Ring: %s -> %s", service_name, next_service_map[service_name])

    # Compute window size for each device: total_layers_per_device / k
    window_sizes: Dict[str, int] = {}
    for service_name, rounds_layers in layer_assignments.items():
        # Flatten to count total layers
        total_layers = sum(len(round_layers) for round_layers in rounds_layers)
        if total_layers > 0:
            window_size = max(1, total_layers // solution_k)
            window_sizes[service_name] = window_size
            logger.info(
                "Window size for %s: %d (total_layers=%d, k=%d)",
                service_name,
                window_size,
                total_layers,
                solution_k,
            )
        else:
            # FIXME: how to handle?
            logger.error(
                "No layers assigned to %s, setting window size to 1",
                service_name,
            )
            window_sizes[service_name] = 1

    logger.info("Layer assignments (by rounds): %s", layer_assignments)
    # return layer_assignments, next_service_map, window_size

    return [
        LayerAssignment(
            service=name,
            layers=layer_assignments[name],
            next_service=next_service_map[name],
            window_size=window_sizes[name],
        )
        for name in device_names
    ]


def optimize_device_ordering(
    shard_profiles: Dict[str, DeviceProfile],
    thunderbolt_conns: Dict[str, Dict[str, ThunderboltConnection]],
) -> list[str]:
    """Optimize device ordering to place Thunderbolt-connected devices adjacently.

    Args:
        shard_profiles: Collected shard profiles
        thunderbolt_conns: Thunderbolt connections mapping (device -> {neighbor -> connection_info})

    Returns:
        Optimized list of device names with head devices first and Thunderbolt neighbors adjacent
    """
    device_names = list(shard_profiles.keys())

    # Find all head devices (multiple shards can run on same machine as API)
    head_devices = []
    for device_name, profile_data in shard_profiles.items():
        if profile_data.is_head:
            head_devices.append(device_name)

    if not head_devices:
        logger.warning("No head device found in profiles, using first device")
        head_devices = [device_names[0]] if device_names else []

    logger.info("Found %d head device(s): %s", len(head_devices), head_devices)

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
    ordered = head_devices.copy()
    remaining = set(device_names) - set(head_devices)

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
