import asyncio
import signal
from socket import gethostname
from secrets import token_hex
from dnet_p2p import AsyncDnetP2P
from dnet.ring.shard.new_shard.adapters.ring import RingAdapter
from dnet.ring.shard.new_shard.runtime import ShardRuntime
from dnet.ring.shard.new_shard.config import TransportConfig
from dnet.ring.shard.new_shard.shard import Shard
from dnet.ring.shard.new_shard.http_api import HTTPServer
from dnet.ring.shard.new_shard.grpc_servicer import GrpcServer
from dnet.utils.logger import logger
from argparse import ArgumentParser

async def serve(grpc_port: int, http_port: int, queue_size: int = 128) -> None:
    shard_id = 1  # In real usage, this would be set via CLI or config
    hostname = gethostname()

    loop = asyncio.get_running_loop()
    discovery = AsyncDnetP2P("lib/dnet-p2p/lib")
    # Core
    runtime = ShardRuntime(shard_id=hostname, queue_size=queue_size)
    adapter = RingAdapter(runtime=runtime, discovery=discovery)
    shard   = Shard(shard_id=shard_id, adapter=adapter)

    # Servers
    grpc_server = GrpcServer(shard=shard, grpc_port=grpc_port)
    http_server = HTTPServer(shard=shard, http_port=http_port, grpc_port=grpc_port, discovery=discovery)

    stop_event = asyncio.Event()

    def _signal_handler(*_: object) -> None:
        logger.warning("Received termination signal. Stopping services.")
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _signal_handler)
    loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    # Start servers first
    await grpc_server.start()
    await http_server.start(stop_event.wait)

    # Start discovery
    # TODO: optionally take shard name from CLI
    instance = f"shard-{token_hex(4)}-{hostname}"
    discovery.create_instance(
        instance,
        http_port,
        grpc_port,
        is_manager=False,  # shard is never a manager
    )
    await discovery.async_start()

    # Finally start shard
    await shard.start(loop)
    await stop_event.wait()
    await shard.shutdown()
    if discovery.is_running():
        await discovery.async_stop()
        await discovery.async_free_instance()
        logger.info(f"Stopped discovery service for node {shard_id}")
    else:
        logger.warning(f"Discovery service for node {shard_id} was not running")
    closed = await http_server.wait_closed(timeout=5.0)
    if not closed:
        await http_server.shutdown()
    await grpc_server.shutdown()


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
