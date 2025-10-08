"""API models for dnet ring topology endpoints."""

import base64
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ..common import TopologyInfo, LayerAssignment


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
# ------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


class ChatBaseParams(BaseModel):
    """Base parameters for chat/completion requests."""

    model: Optional[str] = "default_model"
    temperature: Optional[float] = Field(default=1.0, ge=0)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    repetition_penalty: Optional[float] = Field(default=1.0, ge=0)
    repetition_context_size: Optional[int] = Field(default=20, ge=0)
    logit_bias: Optional[Dict[int, float]] = {}


class ChatParams(ChatBaseParams):
    """Extended parameters for chat requests."""

    stream: Optional[bool] = False
    max_tokens: Optional[int] = Field(default=100, ge=0)
    logit_bias: Optional[Dict[int, float]] = {}
    logprobs: Optional[int] = Field(default=-1)
    stop: Optional[Union[str, List[str]]] = []
    profile: Optional[bool] = False

    def __init__(self, **data: Any):
        super().__init__(**data)
        if isinstance(self.stop, str):
            self.stop = [self.stop]

    @field_validator("logprobs")
    def non_negative_tokens(cls, v: Any) -> Any:
        """Validate logprobs parameter."""
        if v != -1 and not (0 < v <= 10):
            raise ValueError(f"logprobs must be between 1 and 10 but got {v:,}")
        return v


class ChatRequestModel(ChatParams):
    """Request model for chat completions."""

    messages: List[ChatMessage]


class ChatLogProp(BaseModel):
    """Log probabilities for chat completion."""

    token_logprobs: Optional[List[float]] = Field(default=[])
    top_logprobs: Optional[List[Dict[int, float]]] = Field(default=[])
    tokens: Optional[List[int]] = None


class ChatChoice(BaseModel):
    """A single choice in chat completion response."""

    index: int
    message: ChatMessage
    logprop: ChatLogProp
    finish_reason: Optional[ChatCompletionReason] = None


class ChatResponseModel(BaseModel):
    """Response model for chat completions."""

    id: str
    object: str = Field(default="chat.completion")
    model: str = Field(default="default_model")
    choices: List[ChatChoice]
    usage: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


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
    object: str = "text_completion"
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
# Topology Preparation API
# ------------------------


class PrepareTopologyRequest(BaseModel):
    """Request to prepare topology for a model.

    This triggers device discovery, profiling, and optimal layer assignment.
    """

    model: str = Field(..., description="Model name or HuggingFace repo ID")
    force_rediscover: Optional[bool] = Field(
        default=False, description="Force device rediscovery even if cached"
    )


class PrepareTopologyResponse(TopologyInfo):
    pass


# ------------------------
# Model Loading API
# ------------------------


class LoadModelRequest(BaseModel):
    """Request to load model with prepared topology.

    Uses the assignment data from TopologyInfo.
    """

    model: str = Field(..., description="Model name or HuggingFace repo ID")
    assignments: List[LayerAssignment] = Field(
        ..., description="Layer assignments from prepare_topology"
    )


class ShardLoadStatus(BaseModel):
    """Load status for a single shard."""

    service_name: str = Field(..., description="Shard service name")
    success: bool = Field(..., description="Whether loading succeeded")
    layers_loaded: Optional[List[int]] = Field(
        None, description="Layers successfully loaded"
    )
    message: Optional[str] = Field(None, description="Status or error message")


class LoadModelResponse(BaseModel):
    """Response from model loading operation."""

    model: str = Field(..., description="Model name")
    success: bool = Field(..., description="Whether all shards loaded successfully")
    shard_statuses: List[ShardLoadStatus] = Field(
        ..., description="Status of each shard"
    )
    total_load_time_ms: Optional[float] = Field(
        None, description="Total time taken for all loads"
    )
