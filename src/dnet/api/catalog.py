from typing import Any

model_catalog: dict[str, list[dict[str, Any]]] = {
    "models": [
        {
            "id": "mlx-community/gpt-oss-20b-MXFP4-Q8",
            "arch": "gpt_oss",
            "quantization": "8bit",
            "alias": "gpt-oss-20b",
        },
        {
            "id": "mlx-community/gpt-oss-20b-MXFP4-Q4",
            "arch": "gpt_oss",
            "quantization": "4bit",
            "alias": "gpt-oss-20b",
        },
        {
            "id": "mlx-community/gpt-oss-120b-MXFP4-Q8",
            "arch": "gpt_oss",
            "quantization": "8bit",
            "alias": "gpt-oss-120b",
        },
        {
            "id": "mlx-community/gpt-oss-120b-MXFP4-Q4",
            "arch": "gpt_oss",
            "quantization": "4bit",
            "alias": "gpt-oss-120b",
        },
        {
            "id": "Qwen/Qwen3-4B-MLX-bf16",
            "arch": "qwen3",
            "quantization": "bf16",
            "alias": "qwen3-4b",
        },
        {
            "id": "Qwen/Qwen3-4B-MLX-8bit",
            "arch": "qwen3",
            "quantization": "8bit",
            "alias": "qwen3-4b",
        },
        {
            "id": "Qwen/Qwen3-4B-MLX-4bit",
            "arch": "qwen3",
            "quantization": "4bit",
            "alias": "qwen3-4b",
            "ci_test": True,
        },
        {
            "id": "Qwen/Qwen3-8B-MLX-bf16",
            "arch": "qwen3",
            "quantization": "bf16",
            "alias": "qwen3-8b",
        },
        {
            "id": "Qwen/Qwen3-8B-MLX-8bit",
            "arch": "qwen3",
            "quantization": "8bit",
            "alias": "qwen3-8b",
        },
        {
            "id": "Qwen/Qwen3-8B-MLX-4bit",
            "arch": "qwen3",
            "quantization": "4bit",
            "alias": "qwen3-8b",
        },
        {
            "id": "Qwen/Qwen3-14B-MLX-bf16",
            "arch": "qwen3",
            "quantization": "bf16",
            "alias": "qwen3-14b",
        },
        {
            "id": "Qwen/Qwen3-14B-MLX-8bit",
            "arch": "qwen3",
            "quantization": "8bit",
            "alias": "qwen3-14b",
        },
        {
            "id": "Qwen/Qwen3-14B-MLX-4bit",
            "arch": "qwen3",
            "quantization": "4bit",
            "alias": "qwen3-14b",
        },
        {
            "id": "Qwen/Qwen3-32B-MLX-bf16",
            "arch": "qwen3",
            "quantization": "bf16",
            "alias": "qwen3-32b",
        },
        {
            "id": "Qwen/Qwen3-32B-MLX-8bit",
            "arch": "qwen3",
            "quantization": "8bit",
            "alias": "qwen3-32b",
        },
        {
            "id": "Qwen/Qwen3-32B-MLX-4bit",
            "arch": "qwen3",
            "quantization": "4bit",
            "alias": "qwen3-32b",
        },
        {
            "id": "mlx-community/Llama-3.2-3B-Instruct",
            "arch": "llama",
            "quantization": "fp16",
            "alias": "llama-3.2-3b-instruct",
        },
        {
            "id": "mlx-community/Llama-3.2-3B-Instruct-8bit",
            "arch": "llama",
            "quantization": "8bit",
            "alias": "llama-3.2-3b-instruct",
        },
        {
            "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "llama-3.2-3b-instruct",
            "ci_test": True,
        },
        {
            "id": "mlx-community/Llama-3.1-8B-Instruct",
            "arch": "llama",
            "quantization": "fp16",
            "alias": "llama-3.1-8b-instruct",
        },
        {
            "id": "mlx-community/Llama-3.1-8B-Instruct-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "llama-3.1-8b-instruct",
        },
        {
            "id": "mlx-community/llama-3.3-70b-instruct-fp16",
            "arch": "llama",
            "quantization": "fp16",
            "alias": "llama-3.3-70b-instruct",
        },
        {
            "id": "mlx-community/Llama-3.3-70B-Instruct-8bit",
            "arch": "llama",
            "quantization": "8bit",
            "alias": "llama-3.3-70b-instruct",
        },
        {
            "id": "mlx-community/Llama-3.3-70B-Instruct-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "llama-3.3-70b-instruct",
        },
        {
            "id": "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "llama-3.1-70b-instruct",
        },
        {
            "id": "mlx-community/Hermes-4-70B-8bit",
            "arch": "llama",
            "quantization": "8bit",
            "alias": "hermes-4-70b",
        },
        {
            "id": "mlx-community/Hermes-4-70B-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "hermes-4-70b",
        },
        {
            "id": "mlx-community/Hermes-4-405B-4bit",
            "arch": "llama",
            "quantization": "4bit",
            "alias": "hermes-4-405b",
        },
    ]
}


def get_ci_test_models() -> list[dict[str, Any]]:
    """Return models marked for CI integration testing.

    These models are small enough to run on a single CI machine.
    """
    return [m for m in model_catalog["models"] if m.get("ci_test", False)]
