import json
from fastmcp import FastMCP, Context
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from starlette.responses import JSONResponse
from pydantic import ValidationError
from .models import (
    ChatRequestModel,
    ChatMessage,
    APILoadModelRequest,
)
from .inference import InferenceManager
from .model_manager import ModelManager
from .cluster import ClusterManager
from dnet.utils.logger import logger
from .load_helpers import (
    _prepare_topology_core,
    _load_model_core,
    _unload_model_core,
)


class McpError(Exception):
    """Custom MCP error with JSON-RPC 2.0 error codes.
    - -32700: Parse error
    - -32600: Invalid request
    - -32601: Method not found
    - -32602: Invalid params
    - -32603: Internal error
    - -32000 to -32099: Server error (implementation-specific)
    - -32000: Service unavailable (used when no model is loaded)
    - -32001: Request Timeout
    - -32002: Resource not found
    - -32800: Request cancelled
    - -32801: Content too large
    """

    def __init__(self, code: int, message: str, data: dict | None = None):
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(self.message)


def create_mcp_server(
    inference_manager: InferenceManager,
    model_manager: ModelManager,
    cluster_manager: ClusterManager,
) -> FastMCP:
    """Create and configure the MCP server for dnet."""

    mcp = FastMCP("dnet")
    mcp.add_middleware(ErrorHandlingMiddleware())

    @mcp.custom_route("/mcp-health", methods=["GET"])
    async def mcp_health_check(request):
        """Health check endpoint for MCP server."""
        return JSONResponse(
            {
                "status": "healthy",
                "service": "dnet-mcp",
                "model_loaded": model_manager.current_model_id is not None,
                "model": model_manager.current_model_id,
                "topology_configured": cluster_manager.current_topology is not None,
                "shards_discovered": len(cluster_manager.shards)
                if cluster_manager.shards
                else 0,
            }
        )

    @mcp.tool()
    async def chat_completion(
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: str | list[str] | None = None,
        repetition_penalty: float = 1.0,
        ctx: Context | None = None,
    ) -> str:
        """Generate text using distributed LLM inference.
        Args:
            messages: Array of message objects with 'role' and 'content' fields.
                     Each message should be a dict like: {"role": "user", "content": "Hello"}
            model: Model name (optional, uses currently loaded model if not specified)
            temperature: Sampling temperature (0-2), default is 1.0
            max_tokens: Maximum tokens to generate, default is 2000
            top_p: Nucleus sampling parameter (0-1), default is 1.0
            top_k: Top-k sampling parameter (-1 for disabled), default is -1
            stop: Stop sequences (string or list), default is None
            repetition_penalty: Repetition penalty (>=0), default is 1.0
        """

        if ctx:
            await ctx.info("Starting inference...")

        if not model_manager.current_model_id:
            raise McpError(
                -32000,
                "No model loaded. Please load a model first using load_model tool.",
                data={"action": "load_model"},
            )

        model_id = model or model_manager.current_model_id
        stops = [stop] if isinstance(stop, str) else (stop or [])

        try:
            msgs = [
                ChatMessage(**msg) if isinstance(msg, dict) else msg for msg in messages
            ]
            req = ChatRequestModel(
                messages=msgs,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                stop=stops,
                repetition_penalty=repetition_penalty,
                stream=False,
            )
            result = await inference_manager.chat_completions(req)
        except ValidationError as e:
            raise McpError(
                -32602,
                f"Invalid request parameters: {str(e)}",
                data={"validation_errors": str(e)},
            )
        except Exception as e:
            logger.exception("Error in chat_completion: %s", e)
            raise McpError(
                -32603,
                f"Inference failed: {str(e)}",
                data={"model": model_id, "original_error": type(e).__name__},
            )

        if not result.choices or not result.choices[0].message:
            raise McpError(-32603, "No content generated", data={"model": model_id})

        text = result.choices[0].message.content or ""
        if ctx:
            await ctx.info("Inference completed successfully")

        return text

    @mcp.tool()
    async def load_model(
        model: str,
        kv_bits: str = "8bit",
        seq_len: int = 4096,
        ctx: Context | None = None,
    ) -> str:
        """Load a model for distributed inference across the cluster.

        If a different model is already loaded, both models will stay in memory (old model
        is not automatically unloaded). If the same model is already loaded, returns early.
        Automatically prepares topology and discovers devices if needed.

        Args:
            model: Model ID from catalog
            kv_bits: KV cache quantization mode for the model's KV cache, the default is "8bit".
            seq_len: Maximum sequence length (in tokens). defaults to 4096.
        """
        try:
            req = APILoadModelRequest(
                model=model,
                kv_bits=kv_bits,
                seq_len=seq_len,
            )
            if ctx:
                await ctx.info(f"Starting to load model: {req.model}")

            topology = cluster_manager.current_topology
            if topology is None:
                if ctx:
                    await ctx.info("Preparing topology ...")
                try:
                    topology = await _prepare_topology_core(
                        cluster_manager,
                        req.model,
                        req.kv_bits,
                        req.seq_len,
                        progress_callback=ctx.info if ctx else None,
                    )
                except RuntimeError as e:
                    if "No profiles collected" in str(e):
                        raise McpError(
                            -32603,
                            "Failed to collect device profiles. Check shard connectivity.",
                            data={
                                "step": "profiling",
                                "shards_count": len(cluster_manager.shards)
                                if cluster_manager.shards
                                else 0,
                            },
                        )
                    raise
                cluster_manager.current_topology = topology
                if ctx:
                    await ctx.info("Topology prepared")

            if ctx:
                await ctx.info("Loading model layers across shards...")
            response = await _load_model_core(
                cluster_manager,
                model_manager,
                inference_manager,
                topology,
            )

            if not response.success:
                error_msg = response.message or "Model loading failed"
                shard_errors = [
                    {"instance": s.instance, "message": s.message}
                    for s in response.shard_statuses
                    if not s.success
                ]
                raise McpError(
                    -32603,
                    f"Model loading failed: {error_msg}. {len(shard_errors)}/{len(response.shard_statuses)} shards failed.",
                    data={
                        "model": req.model,
                        "shard_errors": shard_errors,
                        "failed_shards": len(shard_errors),
                        "total_shards": len(response.shard_statuses),
                    },
                )

            if ctx:
                await ctx.info(
                    f"Model {req.model} loaded successfully across {len(response.shard_statuses)} shards"
                )

            success_count = len([s for s in response.shard_statuses if s.success])
            return f"Model '{req.model}' loaded successfully. Loaded on {success_count}/{len(response.shard_statuses)} shards."

        except ValidationError as e:
            raise McpError(
                -32602,
                f"Invalid load_model parameters: {str(e)}",
                data={"validation_errors": str(e)},
            )
        except McpError:
            raise
        except Exception as e:
            logger.exception("Error in load_model: %s", e)
            if ctx:
                await ctx.error(f"Failed to load model: {str(e)}")
            raise McpError(
                -32603,
                f"Failed to load model '{model}': {str(e)}",
                data={"model": model, "original_error": type(e).__name__},
            )

    @mcp.tool()
    async def unload_model(ctx: Context | None = None) -> str:
        """Unload the currently loaded model to free memory.
        Unloads the model from all shards and clears the topology. If no model is loaded, returns early.
        """
        if not model_manager.current_model_id:
            return "No model is currently loaded."

        model_name = model_manager.current_model_id
        if ctx:
            await ctx.info(f"Unloading model: {model_name}")

        response = await _unload_model_core(cluster_manager, model_manager)

        if response.success:
            if ctx:
                await ctx.info("Model unloaded successfully")
            return f"Model '{model_name}' unloaded successfully from all shards."
        else:
            shard_errors = [
                {"instance": s.instance, "message": s.message}
                for s in response.shard_statuses
                if not s.success
            ]
            raise McpError(
                -32603,
                "Model unloading failed",
                data={
                    "model": model_name,
                    "shard_errors": shard_errors,
                    "failed_shards": len(shard_errors),
                    "total_shards": len(response.shard_statuses),
                },
            )

    # Resources (for MCP protocol compliance)
    @mcp.resource("mcp://dnet/models")
    def get_available_models() -> str:
        """List of models available in dnet catalog as JSON."""
        return _get_available_models_data()

    @mcp.resource("mcp://dnet/status")
    def get_model_status() -> str:
        """Currently loaded model and cluster status as JSON."""
        return _get_model_status_data()

    @mcp.resource("mcp://dnet/cluster")
    def get_cluster_info() -> str:
        """Cluster information including devices and topology as JSON."""
        return _get_cluster_info_data()

    # Tools that wrap resources (for Claude Desktop compatibility)
    @mcp.tool()
    def list_models() -> str:
        """List all available models in the dnet catalog.

        Returns JSON with model IDs, aliases, and quantization info.
        """
        return _get_available_models_data()

    @mcp.tool()
    def get_status() -> str:
        """Get the current status of dnet.

        Returns JSON with loaded model, topology, and shard count.
        """
        return _get_model_status_data()

    @mcp.tool()
    def get_cluster_details() -> str:
        """Get detailed cluster information.

        Returns JSON with all discovered shards and current topology.
        """
        return _get_cluster_info_data()

    def _get_available_models_data() -> str:
        """Return available models as JSON (same format as /v1/models endpoint)."""
        return json.dumps(
            {
                "object": "list",
                "data": [m.model_dump() for m in model_manager.available_models],
            }
        )

    def _get_model_status_data() -> str:
        """Return current status as JSON."""
        topology = cluster_manager.current_topology
        return json.dumps(
            {
                "model_loaded": model_manager.current_model_id,
                "topology": topology.model_dump() if topology else None,
                "shards_discovered": len(cluster_manager.shards)
                if cluster_manager.shards
                else 0,
            }
        )

    def _get_cluster_info_data() -> str:
        """Return cluster information as JSON (same format as /v1/devices endpoint)."""
        shards = cluster_manager.shards
        topology = cluster_manager.current_topology
        return json.dumps(
            {
                "devices": {
                    name: {
                        "instance": props.instance,
                        "local_ip": props.local_ip,
                        "server_port": props.server_port,
                        "shard_port": props.shard_port,
                        "is_manager": props.is_manager,
                        "is_busy": props.is_busy,
                    }
                    for name, props in shards.items()
                }
                if shards
                else {},
                "topology": topology.model_dump() if topology else None,
            }
        )

    return mcp
