import asyncio
import signal
from socket import gethostname
from secrets import token_hex
from dnet_p2p import AsyncDnetP2P
from dnet.shard.adapters.ring import RingAdapter
from dnet.shard.runtime import ShardRuntime
from dnet.shard.shard import Shard
from dnet.shard.http_api import HTTPServer as ShardHTTPServer
from dnet.shard.grpc_servicer import GrpcServer as ShardGrpcServer
from dnet.utils.logger import logger
from argparse import ArgumentParser


async def serve(
    grpc_port: int,
    http_port: int,
    queue_size: int = 128,
    shard_name: str | None = None,
) -> None:
    shard_id = 1  # In real usage, this would be set via CLI or config
    hostname = gethostname()

    # Resolve instance name: use provided shard_name or fall back to auto-generated
    if shard_name:
        instance_name = shard_name
    else:
        instance_name = f"shard-{token_hex(4)}-{hostname}"

    loop = asyncio.get_running_loop()
    discovery = AsyncDnetP2P("lib/dnet-p2p/lib")
    # Core - use instance_name for runtime to align logs/metrics with discovery name
    runtime = ShardRuntime(shard_id=instance_name, queue_size=queue_size)
    adapter = RingAdapter(runtime=runtime, discovery=discovery)
    shard = Shard(shard_id=shard_id, adapter=adapter)

    # Servers
    grpc_server = ShardGrpcServer(shard=shard, grpc_port=grpc_port)
    http_server = ShardHTTPServer(
        shard=shard, http_port=http_port, grpc_port=grpc_port, discovery=discovery
    )

    stop_event = asyncio.Event()

    def _signal_handler(*_: object) -> None:
        logger.warning("Received termination signal. Stopping services.")
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _signal_handler)
    loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    from dnet.tui import DnetTUI

    tui = DnetTUI(title=f"DNET Shard ({instance_name})")
    tui_task = asyncio.create_task(tui.run(stop_event))

    # Hook into Shard model loading for TUI updates
    original_load_model = shard.load_model
    original_unload_model = shard.unload_model

    async def load_model_wrapper(req):
        tui.update_status(f"Loading model: {req.model_path}...")
        # Initial update with 0 residency until loaded (or we could show target residency)
        residency = int(req.residency_size) if req.residency_size else 0
        tui.update_model_info(
            req.model_path, len(req.layers), residency=residency, loaded=False
        )
        try:
            res = await original_load_model(req)
            tui.update_model_info(
                req.model_path, len(req.layers), residency=residency, loaded=True
            )
            tui.update_status(f"Model loaded: {req.model_path}")
            return res
        except Exception as e:
            tui.update_status(f"Model load failed: {e}")
            raise

    async def unload_model_wrapper():
        tui.update_status("Unloading model...")
        try:
            res = await original_unload_model()
            if res.success:
                tui.update_model_info(None, 0, residency=0, loaded=False)
                tui.update_status("Model unloaded")
            return res
        except Exception as e:
            tui.update_status(f"Model unload failed: {e}")
            raise

    shard.load_model = load_model_wrapper  # type: ignore
    shard.unload_model = unload_model_wrapper  # type: ignore

    try:
        tui.update_status("Starting servers...")
        # Start servers first
        await grpc_server.start()
        await http_server.start(stop_event.wait)

        # Start discovery
        discovery.create_instance(
            instance_name,
            http_port,
            grpc_port,
            is_manager=False,  # shard is never a manager
        )
        logger.info("Registered shard with discovery as '%s'", instance_name)
        await discovery.async_start()

        # Finally start shard
        tui.update_status("Starting shard runtime...")
        await shard.start(loop)

        tui.update_status(f"Shard Running (HTTP:{http_port} gRPC:{grpc_port})")
        await stop_event.wait()
    finally:
        if not tui_task.done():
            tui_task.cancel()
            try:
                await tui_task
            except asyncio.CancelledError:
                pass
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
        help="Activation queue size (default: 256)",
    )
    ap.add_argument(
        "-n",
        "--shard-name",
        type=str,
        default=None,
        help="Custom shard name for discovery registration (default: auto-generated)",
    )
    args = ap.parse_args()

    logger.info(
        "Starting shard server on gRPC port %s, HTTP port %s",
        args.grpc_port,
        args.http_port,
    )
    if args.shard_name:
        logger.info("Using custom shard name: %s", args.shard_name)
    asyncio.run(
        serve(
            args.grpc_port,
            args.http_port,
            args.queue_size,
            shard_name=args.shard_name,
        )
    )


if __name__ == "__main__":
    main()
