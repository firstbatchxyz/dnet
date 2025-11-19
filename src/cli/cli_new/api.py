"""CLI entry point for dnet ring API server (new architecture, decoupled)."""

import asyncio
import signal
from argparse import ArgumentParser
from secrets import token_hex
from socket import gethostname

from dnet.utils.logger import logger
from dnet_p2p import AsyncDnetP2P
from dnet.ring.api.new_api.cluster import ClusterManager
from dnet.ring.api.new_api.model_manager import ModelManager
from dnet.ring.api.new_api.inference import InferenceManager
from dnet.ring.api.new_api.http_api import HTTPServer as ApiHTTPServer
from dnet.ring.api.new_api.grpc_server import GrpcServer as ApiGrpcServer


async def serve(http_port: int, grpc_port: int) -> None:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler(*_: object) -> None:
        logger.warning("Received termination signal. Stopping services.")
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _signal_handler)
    loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    # Discovery
    discovery = AsyncDnetP2P("lib/dnet-p2p/lib")
    node_id = f"api-{token_hex(4)}-{gethostname()}"
    discovery.create_instance(node_id, http_port, grpc_port, is_manager=True)
    await discovery.async_start()

    # Components
    cluster_manager = ClusterManager(discovery)
    model_manager = ModelManager()
    inference_manager = InferenceManager(cluster_manager, model_manager, grpc_port)

    # Servers
    grpc_server = ApiGrpcServer(grpc_port=grpc_port, inference_manager=inference_manager)
    http_server = ApiHTTPServer(
        http_port=http_port,
        cluster_manager=cluster_manager,
        inference_manager=inference_manager,
        model_manager=model_manager,
        node_id=node_id,
    )

    await grpc_server.start()
    await http_server.start(shutdown_trigger=stop_event.wait)

    await stop_event.wait()

    # Shutdown
    closed = await http_server.wait_closed(timeout=5.0)
    if not closed:
        await http_server.shutdown()
    await grpc_server.shutdown()
    if discovery.is_running():
        await discovery.async_stop()
        await discovery.async_free_instance()


def main() -> None:
    ap = ArgumentParser(description="dnet ring API server (new architecture)")
    ap.add_argument("-p", "--http-port", type=int, required=True, help="HTTP server port")
    ap.add_argument("-g", "--grpc-port", type=int, required=True, help="gRPC callback port")
    args = ap.parse_args()

    logger.info(
        f"Starting API server on HTTP port {args.http_port}, gRPC port {args.grpc_port}"
    )
    asyncio.run(serve(args.http_port, args.grpc_port))


if __name__ == "__main__":
    main()
