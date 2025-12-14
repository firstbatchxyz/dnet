from collections import defaultdict
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
from dnet.utils.model import get_model_config_json
from distilp.profiler import profile_model


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
        return JSONResponse({
            "status": "healthy",
            "service": "dnet-mcp",
            "model_loaded": model_manager.current_model_id is not None,
            "model": model_manager.current_model_id,
            "topology_configured": cluster_manager.current_topology is not None,
            "shards_discovered": len(cluster_manager.shards) if cluster_manager.shards else 0,
        })

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
                data={"action": "load_model"}
            )

        model_id = model or model_manager.current_model_id
        stops = [stop] if isinstance(stop, str) else (stop or [])

        try:
            msgs = [
                ChatMessage(**msg) if isinstance(msg, dict) else msg
                for msg in messages
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
                data={"validation_errors": str(e)}
            )
        except Exception as e:
            logger.exception("Error in chat_completion: %s", e)
            raise McpError(
                -32603,
                f"Inference failed: {str(e)}",
                data={"model": model_id, "original_error": type(e).__name__}
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
            kv_bits: KV cache quantization 
            seq_len: Sequence length 
        """
        try:
            req = APILoadModelRequest(
                model=model,
                kv_bits=kv_bits,  
                seq_len=seq_len,
            )
            if ctx:
                await ctx.info(f"Starting to load model: {req.model}")

            if model_manager.current_model_id == req.model:
                return f"Model '{req.model}' is already loaded."

            topology = cluster_manager.current_topology
            if topology is None or topology.model != req.model:
                if ctx:
                    await ctx.info("Preparing topology ...")

                await cluster_manager.scan_devices()
                if not cluster_manager.shards:
                    raise McpError(
                        -32002,
                        "No shards discovered. Check shard connectivity.",
                        data={"action": "check_shard_connectivity"}
                    )

                if ctx:
                    await ctx.info("Profiling cluster performance")

                model_config = get_model_config_json(req.model)
                embedding_size = int(model_config["hidden_size"])
                num_layers = int(model_config["num_hidden_layers"])

                batch_sizes = [1]
                profiles = await cluster_manager.profile_cluster(
                    req.model, embedding_size, 2, batch_sizes
                )
                if not profiles:
                    raise McpError(
                        -32603,
                        "Failed to collect device profiles. Check shard connectivity.",
                        data={
                            "step": "profiling",
                            "shards_count": len(cluster_manager.shards) if cluster_manager.shards else 0
                        }
                    )

                if ctx:
                    await ctx.info("Computing optimal layer distribution")

                model_profile_split = profile_model(
                    repo_id=req.model,
                    batch_sizes=batch_sizes,
                    sequence_length=req.seq_len,
                )
                model_profile = model_profile_split.to_model_profile()

                topology = await cluster_manager.solve_topology(
                    profiles, model_profile, req.model, num_layers, req.kv_bits
                )
                cluster_manager.current_topology = topology

                if ctx:
                    await ctx.info("Topology prepared")

            if ctx:
                await ctx.info("Loading model layers across shards...")
            api_props = await cluster_manager.discovery.async_get_own_properties()
            response = await model_manager.load_model(
                topology, api_props, inference_manager.grpc_port
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
                    f"Model loading failed: {error_msg}. "
                    f"{len(shard_errors)}/{len(response.shard_statuses)} shards failed.",
                    data={
                        "model": req.model,
                        "shard_errors": shard_errors,
                        "failed_shards": len(shard_errors),
                        "total_shards": len(response.shard_statuses)
                    }
                )

            if topology.devices:
                first_shard = topology.devices[0]
                await inference_manager.connect_to_ring(
                    first_shard.local_ip, first_shard.shard_port, api_props.local_ip
                )

            if ctx:
                await ctx.info(f"Model {req.model} loaded successfully across {len(response.shard_statuses)} shards")

            success_count = len([s for s in response.shard_statuses if s.success])
            return f"Model '{req.model}' loaded successfully. Loaded on {success_count}/{len(response.shard_statuses)} shards."

        except ValidationError as e:
            raise McpError(
                -32602,
                f"Invalid load_model parameters: {str(e)}",
                data={"validation_errors": str(e)}
            )
        except McpError:
            raise
        except Exception as e:
            logger.exception("Error in load_model: %s", e)
            if ctx:
                await ctx.error(f"Failed to load model: {str(e)}")
            raise McpError(
                -32603,
                f"Failed to load model '{req.model}': {str(e)}",
                data={"model": req.model, "original_error": type(e).__name__}
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

        await cluster_manager.scan_devices()
        shards = cluster_manager.shards
        response = await model_manager.unload_model(shards)

        if response.success:
            cluster_manager.current_topology = None
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
                    "total_shards": len(response.shard_statuses)
                }
            )

    # Resources (for MCP protocol compliance)
    @mcp.resource("mcp://dnet/models")
    async def get_available_models() -> str:
        """List of models available in dnet catalog, organized by family and quantization."""
        return await _get_available_models_data()

    @mcp.resource("mcp://dnet/status")
    async def get_model_status() -> str:
        """Currently loaded model and cluster status information."""
        return await _get_model_status_data()

    @mcp.resource("mcp://dnet/cluster")
    async def get_cluster_info() -> str:
        """Detailed cluster information including devices and topology."""
        return await _get_cluster_info_data()

    # Tools that wrap resources (for Claude Desktop compatibility)
    @mcp.tool()
    async def list_models() -> str:
        """List all available models in the dnet catalog.

        Returns a formatted list of models organized by family and quantization.
        Use this to see what models you can load.
        """
        return await _get_available_models_data()

    @mcp.tool()
    async def get_status() -> str:
        """Get the current status of dnet including loaded model, topology, and cluster information.

        Returns detailed status about:
        - Currently loaded model (if any)
        - Topology configuration
        - Discovered shards in the cluster
        """
        return await _get_model_status_data()

    @mcp.tool()
    async def get_cluster_details() -> str:
        """Get detailed cluster information including shard details and topology breakdown.

        Returns comprehensive information about:
        - All discovered shards with their IPs and ports
        - Current topology configuration
        - Layer assignments across devices
        """
        return await _get_cluster_info_data()


    async def _get_available_models_data() -> str:
        models_by_family = defaultdict(list)
        for model in model_manager.available_models:
            models_by_family[model.alias].append(model)

        output_lines = ["Available Models in dnet Catalog:\n"]
        output_lines.append("=" * 60)

        for family_name in sorted(models_by_family.keys()):
            models = sorted(models_by_family[family_name], key=lambda m: m.id)
            output_lines.append(f"\n{family_name.upper()}")
            output_lines.append("-" * 60)

            by_quant = defaultdict(list)
            for model in models:
                by_quant[model.quantization].append(model)

            for quant in ["bf16", "fp16", "8bit", "4bit"]:
                if quant in by_quant:
                    quant_models = by_quant[quant]
                    quant_display = {
                        "bf16": "BF16 (Full precision)",
                        "fp16": "FP16 (Full precision)",
                        "8bit": "8-bit quantized",
                        "4bit": "4-bit quantized (smallest)",
                    }.get(quant, quant)
                    output_lines.append(f"  {quant_display}:")
                    for model in quant_models:
                        output_lines.append(f"    - {model.id}")

        output_lines.append("\n" + "=" * 60)
        output_lines.append(f"\nTotal: {len(model_manager.available_models)} models")
        output_lines.append("\nTo load a model, use the load_model tool with the full model ID.")

        return "\n".join(output_lines)

    async def _get_model_status_data() -> str:
        status_lines = ["dnet Status"]
        status_lines.append("=" * 60)

        if model_manager.current_model_id:
            status_lines.append(f"\n Model Loaded: {model_manager.current_model_id}")
        else:
            status_lines.append("\n No Model Loaded")

        topology = cluster_manager.current_topology
        if topology:
            status_lines.append(f"\n Topology:\n  Model: {topology.model}\n  Devices: {len(topology.devices)}\n  Layers: {topology.num_layers}\n  KV Cache: {topology.kv_bits}")

            if topology.assignments:
                status_lines.append(f"\n  Layer Distribution:")
                for assignment in topology.assignments:
                    layers_str = ", ".join(
                        f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0])
                        for r in assignment.layers
                    )
                    status_lines.append(
                        f"    {assignment.instance}: layers [{layers_str}]"
                    )
        else:
            status_lines.append("\n Topology: Not configured")

        shards = cluster_manager.shards
        if shards:
            shard_names = ", ".join(sorted(shards.keys()))
            status_lines.append(f"\n  Cluster:\n  Discovered Shards: {len(shards)}\n  Shard Names: {shard_names}")
        else:
            status_lines.append("\n  Cluster: No shards discovered")

        status_lines.append("\n" + "=" * 60)

        return "\n".join(status_lines)

    async def _get_cluster_info_data() -> str:
        output_lines = ["dnet Cluster Information"]
        output_lines.append("=" * 60)

        shards = cluster_manager.shards
        if shards:
            output_lines.append(f"\n  Shards ({len(shards)}):")
            for name, props in sorted(shards.items()):
                output_lines.append(f"\n  {name}:\n    IP: {props.local_ip}\n    HTTP Port: {props.server_port}\n    gRPC Port: {props.shard_port}\n    Manager: {'Yes' if props.is_manager else 'No'}\n    Busy: {'Yes' if props.is_busy else 'No'}")
        else:
            output_lines.append("\n  No shards discovered")

        topology = cluster_manager.current_topology
        if topology:
            output_lines.append(f"\n Topology:\n  Model: {topology.model}\n  Total Layers: {topology.num_layers}\n  KV Cache Bits: {topology.kv_bits}\n  Devices: {len(topology.devices)}")

            if topology.assignments:
                output_lines.append(f"\n  Layer Assignments:")
                for assignment in topology.assignments:
                    layers_flat = [
                        layer
                        for round_layers in assignment.layers
                        for layer in round_layers
                    ]
                    layers_str = ", ".join(map(str, sorted(layers_flat)))
                    output_lines.append(
                        f"    {assignment.instance}: [{layers_str}] "
                        f"(window={assignment.window_size}, "
                        f"next={assignment.next_instance or 'N/A'})"
                    )
        else:
            output_lines.append("\n No topology configured")
        
        output_lines.append("\n" + "=" * 60)

        return "\n".join(output_lines)

    return mcp
