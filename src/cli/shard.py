"""CLI entry point for dnet ring shard server."""

import asyncio
import random
import signal
from argparse import ArgumentParser

from dnet.utils.logger import logger
from dnet.ring.shard import RingShardNode


async def serve(
    grpc_port: int,
    http_port: int,
    queue_size: int = 128,
) -> None:
    """Serve the shard node.

    Args:
        grpc_port: gRPC server port
        http_port: HTTP server port
        queue_size: Activation queue size
    """
    node_id = random.randint(0, 1000)
    shard_node = RingShardNode(
        node_id=node_id,
        grpc_port=grpc_port,
        http_port=http_port,
        queue_size=queue_size,
    )

    # Handle shutdown signals gracefully
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler(*_: object) -> None:
        logger.warning("Received termination signal. Stopping services.")
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _signal_handler)
    loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    await shard_node.start(shutdown_trigger=stop_event.wait)
    await stop_event.wait()
    await shard_node.shutdown()


def main() -> None:
    """Run dnet ring shard server.

    The shard server runs without a preloaded model. The API will send
    LoadModel commands via HTTP to configure which layers to load.
    """
    ap = ArgumentParser(description="dnet ring shard server")
    ap.add_argument(
        "-p",
        "--grpc-port",
        type=int,
        required=True,
        help="gRPC server port",
    )
    ap.add_argument(
        "--http-port",
        type=int,
        required=True,
        help="HTTP server port",
    )
    ap.add_argument(
        "-q",
        "--queue-size",
        type=int,
        default=256,
        help="Activation queue size (default: 10)",
    )
    args = ap.parse_args()

    logger.info(
        "Starting shard server on gRPC port %s, HTTP port %s",
        args.grpc_port,
        args.http_port,
    )
    asyncio.run(
        serve(
            args.grpc_port,
            args.http_port,
            args.queue_size,
        )
    )


if __name__ == "__main__":
    main()
