"""Base class for ring topology models."""

from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class BaseRingModel(nn.Module, metaclass=ABCMeta):
    """Base class for models used in ring topology.

    Subclasses must implement embedding, normalization, LM projection,
    and layer-by-layer application for distributed inference.
    """

    model_type: Optional[str] = None

    @abstractmethod
    def embed(self, x: mx.array) -> mx.array:
        """Embed input tokens.

        Args:
            x: Input token IDs

        Returns:
            Embedded representations
        """

    @abstractmethod
    def normalize(self, x: mx.array) -> mx.array:
        """Apply final normalization.

        Args:
            x: Hidden states

        Returns:
            Normalized hidden states
        """

    @abstractmethod
    def lm_project(self, x: mx.array) -> mx.array:
        """Project to vocabulary logits.

        Args:
            x: Normalized hidden states

        Returns:
            Logits over vocabulary
        """

    @abstractmethod
    def forward(self, x: mx.array, cache: Optional[Any] = None) -> mx.array:
        """Forward pass through the model.

        Args:
            x: Input token IDs or embeddings
            cache: Optional KV cache

        Returns:
            Model output
        """

    @abstractmethod
    def apply_single_layer(
        self, layer_idx: int, x: mx.array, cache: Optional[Any] = None
    ) -> mx.array:
        """Apply a single decoding layer by absolute index.

        Implementations should map `layer_idx` to their local layers if they
        only host a subset, and use the correct per-layer cache entry if
        provided.

        Args:
            layer_idx: Absolute layer index
            x: Layer input
            cache: Optional per-layer cache

        Returns:
            Layer output
        """

    @property
    @abstractmethod
    def decoding_layers(self) -> Any:
        """Get the decoding layers.

        Returns:
            Layer container (e.g., ModuleList)
        """

    @property
    @abstractmethod
    def head_dim(self) -> Tuple[int, int]:
        """Get head dimensions.

        Returns:
            Tuple of (num_heads, dim_per_head)
        """

    @property
    @abstractmethod
    def n_kv_heads(self) -> int:
        """Get number of key/value heads.

        Returns:
            Number of KV heads
        """

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Get total number of layers in the model.

        Returns:
            Number of layers
        """
