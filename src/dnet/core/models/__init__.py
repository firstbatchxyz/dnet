"""Model implementations for ring topology."""

from typing import Any, List, Optional

from dnet.utils.loader import subclass_where
from .base import BaseRingModel
from .deepseek_v2 import DeepseekV2RingModel
from .llama import LlamaRingModel
from .gpt_oss import GptOssRingModel
from .qwen3 import Qwen3RingModel


def get_ring_model(
    model_type: str,
    model_config: Any,
    assigned_layers: Optional[List[int]] = None,
    is_api_layer: bool = False,
) -> BaseRingModel:
    """Get ring model instance by type.

    Args:
        model_type: Model type identifier
        model_config: Model configuration
        assigned_layers: Assigned layer indices
        is_api_layer: Whether this is an API layer

    Returns:
        Ring model instance
    """
    cls = subclass_where(BaseRingModel, model_type=model_type)
    return cls(
        model_config=model_config,
        assigned_layers=assigned_layers,
        is_api_layer=is_api_layer,
    )


__all__ = [
    "BaseRingModel",
    "DeepseekV2RingModel",
    "LlamaRingModel",
    "GptOssRingModel",
    "Qwen3RingModel",
    "get_ring_model",
]
