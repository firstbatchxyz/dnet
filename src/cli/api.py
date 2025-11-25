"""CLI entry point for dnet ring API server (new architecture, decoupled)."""

import asyncio
import signal
from argparse import ArgumentParser
from secrets import token_hex
from socket import gethostname

from dnet.utils.logger import logger
from dnet_p2p import AsyncDnetP2P
from dnet.api.cluster import ClusterManager
from dnet.api.model_manager import ModelManager
from dnet.api.inference import InferenceManager
from dnet.api.http_api import HTTPServer as ApiHTTPServer
from dnet.api.grpc_servicer import GrpcServer as ApiGrpcServer


async def serve(http_port: int, grpc_port: int) -> None:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler(*_: object) -> None:
        logger.warning("Received termination signal. Stopping services.")
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _signal_handler)
    loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    # print_startup_banner(tag="api")

    # TUI Setup
    from dnet.tui import DnetTUI

    tui = DnetTUI(title="DNET API Server")
    tui_task = asyncio.create_task(tui.run(stop_event))

    try:
        tui.update_status("Initializing Discovery...")
        # Discovery
        discovery = AsyncDnetP2P("lib/dnet-p2p/lib")
        node_id = f"api-{token_hex(4)}-{gethostname()}"
        discovery.create_instance(node_id, http_port, grpc_port, is_manager=True)
        await discovery.async_start()

        # Components
        from dnet.api.strategies.ring import RingStrategy
        from typing import Optional

        strategy = RingStrategy()  # ContextParallelStrategy()

        def update_tui_model_info(
            model_name: Optional[str], layers: int, loaded: bool
        ) -> None:
            if not model_name:
                tui.update_model_info("", 0, 0, False, show_layers_visual=False)
            else:
                tui.update_model_info(
                    model_name,
                    layers,
                    residency=0,
                    loaded=loaded,
                    show_layers_visual=False,
                )

        cluster_manager = ClusterManager(discovery, solver=strategy.solver)
        model_manager = ModelManager(on_model_change=update_tui_model_info)
        inference_manager = InferenceManager(
            cluster_manager, model_manager, grpc_port, adapter=strategy.adapter
        )

        # Servers
        grpc_server = ApiGrpcServer(
            grpc_port=grpc_port, inference_manager=inference_manager
        )
        http_server = ApiHTTPServer(
            http_port=http_port,
            cluster_manager=cluster_manager,
            inference_manager=inference_manager,
            model_manager=model_manager,
            node_id=node_id,
        )

        tui.update_status("Starting Servers...")
        await grpc_server.start()
        await http_server.start(shutdown_trigger=stop_event.wait)

        tui.update_status(f"Running on HTTP:{http_port} gRPC:{grpc_port}")
        await stop_event.wait()
    finally:
        # Ensure TUI task is cancelled or finished if we exit early
        if not tui_task.done():
            tui_task.cancel()
            try:
                await tui_task
            except asyncio.CancelledError:
                pass

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
    ap.add_argument(
        "-p", "--http-port", type=int, required=True, help="HTTP server port"
    )
    ap.add_argument(
        "-g", "--grpc-port", type=int, required=True, help="gRPC callback port"
    )
    args = ap.parse_args()

    logger.info(
        f"Starting API server on HTTP port {args.http_port}, gRPC port {args.grpc_port}"
    )
    asyncio.run(serve(args.http_port, args.grpc_port))


if __name__ == "__main__":
    main()
