"""Centralized configuration using Pydantic Settings with .env support.

Usage:
    from dnet.config import get_settings
    settings = get_settings()
    print(settings.api.http_port)

Environment variables are loaded from:
1. .env file in the project root
2. System environment variables (override .env)
3. CLI arguments (override env vars when passed to CLIs)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="DNET_")

    log: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    profile: bool = Field(
        default=False,
        description="Enable profile logging",
    )


class ObservabilitySettings(BaseSettings):
    """Observability/profiling configuration."""

    model_config = SettingsConfigDict(env_prefix="DNET_OBS_")

    enabled: bool = Field(
        default=False,
        description="Enable profiling (also enabled by DNET_PROFILE=true)",
    )
    sync_per_layer: bool = Field(
        default=False,
        description="Sync after each layer computation",
    )
    sync_every_n: int = Field(
        default=0,
        description="Sync every N iterations (0=disabled)",
    )


class KVCacheSettings(BaseSettings):
    """KV cache configuration."""

    model_config = SettingsConfigDict(env_prefix="DNET_KV_")

    mode: Literal["fp16", "8bit", "4bit", "quant"] = Field(
        default="8bit",
        description="Cache quantization mode",
    )
    bits: int = Field(
        default=8,
        description="Bits for quant mode",
    )
    group_size: int = Field(
        default=64,
        description="Quantization group size",
    )
    ttl_s: float = Field(
        default=30.0,
        description="KV cache TTL in seconds",
    )


class ComputeSettings(BaseSettings):
    """Compute settings for shard."""

    model_config = SettingsConfigDict(env_prefix="DNET_COMPUTE_")

    prefetch_mode: str = Field(
        default="off",
        description="Prefetch mode for layer loading (off, sync, async)",
    )
    mxload_fastpath: bool = Field(
        default=True,
        description="Use mlx fast load path",
    )
    input_pool_mb: int = Field(
        default=512,
        description="Input memory pool MB",
    )
    output_pool_mb: int = Field(
        default=512,
        description="Output memory pool MB",
    )


class TransportSettings(BaseSettings):
    """Transport settings for shard."""

    model_config = SettingsConfigDict(env_prefix="DNET_TRANSPORT_")

    wire_dtype: str = Field(
        default="fp16",
        description="Wire dtype for activations",
    )
    streaming: bool = Field(
        default=True,
        description="Enable streaming transport",
    )
    stream_backoff_s: float = Field(
        default=0.5,
        description="Stream backoff seconds",
    )
    stream_idle_s: float = Field(
        default=2.0,
        description="Stream idle timeout seconds",
    )
    send_retries: int = Field(
        default=3,
        description="Number of send retries",
    )
    compress: bool = Field(
        default=False,
        description="Enable compression",
    )
    compress_min_bytes: int = Field(
        default=65536,
        description="Min bytes for compression",
    )


class GrpcSettings(BaseSettings):
    """gRPC settings."""

    model_config = SettingsConfigDict(env_prefix="DNET_GRPC_")

    max_message_length: int = Field(
        default=64 * 1024 * 1024,
        description="Max gRPC message length",
    )
    max_concurrent_streams: int = Field(
        default=1024,
        description="Max concurrent streams",
    )
    keepalive_time_ms: int = Field(
        default=120000,
        description="Keepalive interval ms",
    )
    keepalive_timeout_ms: int = Field(
        default=20000,
        description="Keepalive timeout ms",
    )


class StorageSettings(BaseSettings):
    """Storage paths configuration."""

    model_config = SettingsConfigDict(env_prefix="DNET_")

    repack_dir: str = Field(
        default="~/.dria/dnet/repacked_layers",
        description="Repacked layers directory",
    )
    log_dir: str = Field(
        default="~/.dria/dnet/logs",
        description="Log files directory",
    )


class ApiSettings(BaseSettings):
    """API server settings."""

    model_config = SettingsConfigDict(env_prefix="DNET_API_")

    http_port: int = Field(
        default=8080,
        description="HTTP server port",
    )
    grpc_port: int = Field(
        default=50051,
        description="gRPC callback port",
    )
    compression_pct: float = Field(
        default=0.0,
        description="Compression percentage",
    )
    max_concurrent_requests: int = Field(
        default=100,
        description="Max concurrent requests",
    )
    discovery_port: int = Field(
        default=0,
        description="Discovery port (0=dynamic)",
    )


class ShardSettings(BaseSettings):
    """Shard server settings."""

    model_config = SettingsConfigDict(env_prefix="DNET_SHARD_")

    http_port: int = Field(
        default=8081,
        description="HTTP server port",
    )
    grpc_port: int = Field(
        default=50052,
        description="gRPC server port",
    )
    queue_size: int = Field(
        default=256,
        description="Activation queue size",
    )
    name: str | None = Field(
        default=None,
        description="Custom shard name",
    )


class TopologySettings(BaseSettings):
    """Topology configuration for shard."""

    model_config = SettingsConfigDict(env_prefix="DNET_TOPOLOGY_")

    resident_windows: int = Field(
        default=1,
        description="Number of resident windows",
    )
    warmup_windows: int = Field(
        default=1,
        description="Number of warmup windows",
    )
    x_stats: bool = Field(
        default=False,
        description="Enable extended statistics",
    )


class DnetSettings(BaseSettings):
    """Main dnet settings, loads from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested settings (each reads its own env vars)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    shard: ShardSettings = Field(default_factory=ShardSettings)
    transport: TransportSettings = Field(default_factory=TransportSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    kv_cache: KVCacheSettings = Field(default_factory=KVCacheSettings)
    grpc: GrpcSettings = Field(default_factory=GrpcSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    topology: TopologySettings = Field(default_factory=TopologySettings)


@lru_cache
def get_settings() -> DnetSettings:
    """Get cached settings instance."""
    return DnetSettings()


# Export all settings classes for introspection (used by generate_env_example.py)
__all__ = [
    "DnetSettings",
    "get_settings",
    "LoggingSettings",
    "ObservabilitySettings",
    "ApiSettings",
    "ShardSettings",
    "TransportSettings",
    "ComputeSettings",
    "KVCacheSettings",
    "GrpcSettings",
    "StorageSettings",
    "TopologySettings",
]
