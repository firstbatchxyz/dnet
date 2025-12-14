#!/usr/bin/env python3
"""Generate .env.example from DnetSettings fields.

This script introspects the Pydantic settings classes and generates
a documented .env.example file with all available environment variables.

Usage:
    uv run python scripts/generate_env_example.py
"""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    """Generate .env.example from settings definitions."""
    from dnet.config import (
        ApiSettings,
        ComputeSettings,
        GrpcSettings,
        KVCacheSettings,
        LoggingSettings,
        ObservabilitySettings,
        ShardSettings,
        StorageSettings,
        TopologySettings,
        TransportSettings,
    )

    lines = [
        "# Dnet Configuration",
        "# Auto-generated from settings definitions - DO NOT EDIT MANUALLY",
        "# Copy to .env and modify as needed",
        "",
    ]

    # Define sections with their settings classes
    # Order matters for readability
    settings_sections = [
        ("Logging", LoggingSettings),
        ("Observability / Profiling", ObservabilitySettings),
        ("API Server", ApiSettings),
        ("Shard Server", ShardSettings),
        ("Topology", TopologySettings),
        ("Transport", TransportSettings),
        ("Compute", ComputeSettings),
        ("KV Cache", KVCacheSettings),
        ("gRPC", GrpcSettings),
        ("Storage", StorageSettings),
    ]

    for section_name, cls in settings_sections:
        lines.append(f"# === {section_name} ===")

        # Get the env_prefix from model_config
        model_config = getattr(cls, "model_config", {})
        prefix = model_config.get("env_prefix", "")

        for field_name, field_info in cls.model_fields.items():  # type: ignore[attr-defined]
            env_var = f"{prefix}{field_name.upper()}"
            default = field_info.default
            desc = field_info.description or ""

            # Format value based on type
            if isinstance(default, bool):
                val = "true" if default else "false"
            elif default is None:
                val = ""
            else:
                val = str(default)

            # Add description as comment
            if desc:
                lines.append(f"# {desc}")

            lines.append(f"{env_var}={val}")

        lines.append("")

    # Write to .env.example
    output = Path(__file__).parent.parent / ".env.example"
    output.write_text("\n".join(lines))
    print(f"Generated {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
