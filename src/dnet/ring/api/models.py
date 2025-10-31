"""API models for dnet ring topology endpoints."""

import base64
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ..common import LayerAssignment


class RoleMapping(BaseModel):
    """Role mapping for chat formatting."""

    system_prompt: str = Field(
        default="A chat between a curious user and an artificial intelligence "
        "assistant. The assistant follows the given rules no matter what."
    )
    system: str = Field(default="ASSISTANT's RULE: ")
    user: str = Field(default="USER: ")
    assistant: str = Field(default="ASSISTANT: Always respond in English")
    stop: str = Field(default="\n")


class ChatCompletionReason(str, Enum):
    """Reason for chat completion ending."""

    LENGTH = "length"
    STOP = "stop"


class RingInferenceError(BaseModel):
    """Error response for ring inference."""

    error: str
    error_code: str
    nonce: str
    failed_node: Optional[str] = None
    traceback: Optional[str] = None


# ------------------------
# Chat API
# OpenAI compatible:
# - https://platform.openai.com/docs/api-reference/chat
# - https://platform.openai.com/docs/api-reference/chat/object
# -
# ------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str  # "system" | "user" | "assistant" | "tool" | "developer" # TODO: use Literal?
    content: str


class ChatParams(BaseModel):
    """Parameters for chat requests, used by:
    - `ChatRequestModel` with messages
    - `CompletionRequestModel` with prompt

    The attributes are given in the order they appear on docs, with unused ones commented out.
    """

    # messages / prompt
    model: str
    # audio: Optional[ChatAudioParams] = Field(default=None)  # NOTE: unused
    # frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)  # NOTE: unused
    # logit_bias: Optional[Dict[int, float]] = Field(default=None) # NOTE: unused
    logprobs: Optional[bool] = Field(default=False)
    max_tokens: int = Field(
        default=100, ge=0
    )  # NOT using `max_completion_tokens` because that is for `O-` models only
    # metadata: Optional[Dict[str, Any]] = Field(default=None)  # NOTE: unused, we dont store metadata
    # modalities: Optional[List[str]] = Field(default=None)  # NOTE: unused
    # parallel_tool_calls: bool = Field(default=False)  # NOTE: unused
    # prediction: NOT USED
    # presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)  # NOTE: unused
    # prompt_cache_key: Optional[str] = Field(default=None)  # NOTE: unused
    # TODO: response_format:
    # safety_identifier: Optional[str] = Field(default=None)  # NOTE: unused
    # service_tier: Optional[str] = Field(default=None)  # NOTE: unused
    stop: Union[str, List[str]] = Field(default_factory=list)
    # store: bool = Field(default=False)  # NOTE: unused
    stream: bool = Field(default=False)
    # stream_options:  # NOTE: unused
    temperature: float = Field(default=1.0, ge=0, le=2)
    # tool_choice: # NOTE: unused, later with tool calling
    # tools: # NOTE: unused, later with tool calling
    top_logprobs: int = Field(default=0, ge=0, le=20)
    top_p: float = Field(default=1.0, ge=0, le=1)
    verbosity: Literal["low", "medium", "high"] = Field(default="medium")  # TODO: used?
    # web_search_options:  # NOTE: unused, later with web search

    ## NON-OPENAI PARAMETERS ##
    repetition_penalty: float = Field(default=1.0, ge=0)  # FIXME: what is this?
    repetition_context_size: int = Field(default=20, ge=0)  # FIXME: what is this?
    profile: bool = Field(default=False)

    def __init__(self, **data: Any):
        super().__init__(**data)

        # FIXME: why do this?
        if isinstance(self.stop, str):
            self.stop = [self.stop]

    @field_validator("logprobs")
    def non_negative_tokens(cls, v: Any) -> Any:
        """Validate logprobs parameter."""
        if v != -1 and not (0 < v <= 10):
            raise ValueError(f"logprobs must be between 1 and 10 but got {v:,}")
        return v


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatRequestModel(ChatParams):
    """Request model for chat completions."""

    messages: List[ChatMessage]


class ChatLogProbs(BaseModel):
    """Log probabilities for chat completion."""

    content: None = Field(default=None)
    message: None = Field(default=None)

    token_logprobs: Optional[List[float]] = Field(default_factory=list)
    top_logprobs: Optional[List[Dict[int, float]]] = Field(default_factory=list)
    tokens: Optional[List[int]] = None


class ChatChoice(BaseModel):
    """A single choice in chat completion response."""

    index: int
    message: ChatMessage
    logprobs: ChatLogProbs
    finish_reason: Optional[ChatCompletionReason] = None


class ChatResponseModel(BaseModel):
    """Response model for chat completions.

    Compatible with [OpenAI](https://platform.openai.com/docs/api-reference/chat/object), except the following fields:

    -"""

    choices: List[ChatChoice] = Field(
        ..., description="List of chat completion choices."
    )
    created: int = Field(..., description="The Unix timestamp (in seconds) of when the chat completion was created.")  # fmt: skip
    id: str = Field(..., description="Unique identifier for the chat completion.")
    model: str = Field(default="default_model", description="The model used for the chat completion.")  # fmt: skip
    object: Literal["chat.completion"] = "chat.completion"
    usage: Optional[ChatUsage] = None
    metrics: Optional[Dict[str, Any]] = None


# ------------------------
# Embeddings API
# ------------------------


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingObject(BaseModel):
    """Response model for embeddings."""

    object: Literal["embedding"] = "embedding"
    data: List[Dict[str, Any]] = Field(
        ..., description="List of embedding data objects."
    )
    model: str = Field(..., description="Model name or HuggingFace repo ID.")


class EmbeddingRequestModel(BaseModel):
    input: Union[str, List[str]] = Field(
        ..., description="Input text or list of texts to embed."
    )
    model: str = Field(..., description="Model name or HuggingFace repo ID.")
    # dimensions:  # NOTE: unused
    encoding_format: Literal["base64", "float32"] = Field(
        default="float32", description="Encoding format for the embeddings."
    )
    # user:  # NOTE: unused


class EmbeddingResponseModel(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Optional[EmbeddingsUsage] = Field(default=None)


# ------------------------
# Completion API
# ------------------------


class CompletionRequestModel(ChatParams):
    """Request model for text completions."""

    prompt: str


class CompletionChoice(BaseModel):
    """A single choice in completion response."""

    index: int
    text: str
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class CompletionResponseModel(BaseModel):
    """Response model for text completions."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Dict[str, Any]] = None


# ------------------------
# Response API
# ------------------------


class RecieveResultRequest(BaseModel):
    """Request to receive computation result."""

    nonce: str
    batch_size: int
    shape: Tuple[int, ...]
    dtype: str
    layer_id: int
    timestamp: int
    node_origin: str
    data: str

    @staticmethod
    def encode(data: bytes) -> str:
        """Encode bytes to base64 string."""
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def decode(data: str) -> bytes:
        """Decode base64 string to bytes."""
        return base64.b64decode(data)


class RecieveResultResponseModel(JSONResponse):
    """Response for result receipt."""
    pass


# FIXME: can we do better?
RecieveResultRequestModel = Union[RecieveResultRequest, RingInferenceError]

# ------------------------
# /v1/models
# ------------------------


class ModelObject(BaseModel):
    created: int  # file creation timestamp
    id: str  # repo id
    object: Literal["model"] = "model"
    owned_by: Literal["local"] = "local"  # TODO: unless we can get the model owner name


type ListModelsResponseModel = list[ModelObject]
type RetrieveModelResponseModel = ModelObject


# ------------------------
# Topology Preparation API
# ------------------------


class PrepareTopologyRequest(BaseModel):
    """Request to prepare topology for a model.

    This triggers device discovery, profiling, and optimal layer assignment.
    """

    model: str = Field(..., description="Model name or HuggingFace repo ID")
    kv_bits: Literal["4bit", "8bit", "fp16"] = Field(default="4bit", description="KV cache quantization")
    seq_len: int = Field(default=256, description="Sequence length to optimize for")
    max_batch_exp: int = Field(
        default=2, description="Max batch size as power of 2 exponent"
    )


class ManualDevice(BaseModel):
    """Manual device specification for topology (no discovery)."""

    instance: str = Field(..., description="Name of the device")
    local_ip: str = Field(..., description="Reachable IP/host for the device")
    server_port: int = Field(..., description="Device HTTP port (for /load_model)")
    shard_port: int = Field(..., description="Device gRPC port (ring service)")


class PrepareTopologyManualRequest(BaseModel):
    """Prepare topology manually by providing devices and assignments.

    This bypasses discovery and profiling. The resulting topology is stored and
    can be used with /v1/load_model.
    """

    model: str = Field(..., description="Model name or HuggingFace repo ID")
    kv_bits: Literal["4bit", "8bit", "fp16"] = Field(
        default="8bit", description="KV cache quantization to use"
    )
    # FIXME: can use DnetDeviceProperties instead?
    devices: List[ManualDevice] = Field(..., description="Manual device endpoints")
    assignments: List[LayerAssignment] = Field(
        ..., description="Layer assignments per device (rounds or flat)"
    )
    num_layers: Optional[int] = Field(
        default=None,
        description="Total number of layers (optional; inferred if missing)",
    )


class APILoadModelRequest(BaseModel):
    """Request to load model with prepared topology.

    Uses the assignment data from TopologyInfo.
    """

    model: Optional[str] = Field(
        default=None, description="Model name or HuggingFace repo ID (optional)"
    )
    kv_bits: Literal["4bit", "8bit", "fp16"] = Field(
        default="8bit",
        description="KV cache quantization level"
    )
    seq_len: int = Field(
        default=512,
        description="Sequence length to optimize for. Select this based on your use case."
    )
    batch_size: int = Field(
        default=1,
        description="Batch size to optimize for. Select this based on your use case."
    )


# FIXME: move elsewhere
class ShardLoadStatus(BaseModel):
    """Load status for a single shard."""

    instance: str = Field(..., description="Shard name")
    success: bool = Field(..., description="Whether loading succeeded")
    layers_loaded: Optional[List[int]] = Field(
        None, description="Layers successfully loaded"
    )
    message: Optional[str] = Field(None, description="Status or error message")


class APILoadModelResponse(BaseModel):
    """Response from model loading operation."""

    model: str = Field(..., description="Model name")
    success: bool = Field(..., description="Whether all shards loaded successfully")
    shard_statuses: List[ShardLoadStatus] = Field(
        ..., description="Status of each shard"
    )
    message: Optional[str] = Field(
        default=None, description="Overall status or error message"
    )


# FIXME: move elsewhere
class ShardUnloadStatus(BaseModel):
    """Unload status for a single shard."""

    instance: str = Field(..., description="Shard name")
    success: bool = Field(..., description="Whether unloading succeeded")
    message: Optional[str] = Field(None, description="Status or error message")


class UnloadModelResponse(BaseModel):
    """Response from model unloading operation."""

    success: bool = Field(..., description="Whether all shards unloaded successfully")
    shard_statuses: List[ShardUnloadStatus] = Field(
        ..., description="Status of each shard"
    )
    message: Optional[str] = Field(
        default=None, description="Overall status or error message"
    )
