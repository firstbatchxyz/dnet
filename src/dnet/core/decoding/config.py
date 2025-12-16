from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class DecodingConfig:
    """Configuration for decoding/sampling strategy."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled
    repetition_penalty: float = 1.0
    logit_bias: dict[int, float] | None = None
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    # Structured output support
    grammar_json_schema: Optional[str] = None  # JSON schema string for grammar-constrained generation
