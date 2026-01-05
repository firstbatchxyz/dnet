"""End-to-end tests for structured outputs functionality.

Usage (with servers running):
    uv run pytest tests/integration/test_structured_outputs_e2e.py -v

"""

import json
import logging
import time
from typing import Any, Generator

import pytest
import requests

from dnet.api.catalog import get_ci_test_models

logger = logging.getLogger(__name__)

# Server configuration
API_HTTP_PORT = 8080
BASE_URL = f"http://localhost:{API_HTTP_PORT}"

# Timeouts
HEALTH_CHECK_TIMEOUT = 30  # seconds to wait for server health
INFERENCE_TIMEOUT = 60  # seconds for inference requests
MODEL_LOAD_TIMEOUT = 300  # seconds to wait for model loading

# Get CI-testable models from catalog
CI_TEST_MODELS = get_ci_test_models()

# Find the Qwen 4B 4bit model for structured outputs testing
STRUCTURED_OUTPUTS_MODEL = None
for model in CI_TEST_MODELS:
    if model["id"] == "Qwen/Qwen3-4B-MLX-4bit":
        STRUCTURED_OUTPUTS_MODEL = model
        break

# Fallback if not found
if STRUCTURED_OUTPUTS_MODEL is None:
    STRUCTURED_OUTPUTS_MODEL = {"id": "Qwen/Qwen3-4B-MLX-4bit", "alias": "qwen3-4b"}


def wait_for_health(url: str, timeout: float = HEALTH_CHECK_TIMEOUT) -> bool:
    """Wait for server health endpoint to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except (requests.RequestException, requests.ConnectionError):
            time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def servers() -> Generator[None, None, None]:
    """Check that API server is healthy (assumes servers are already running)."""
    # Assume servers are already running (like CI does)
    if not wait_for_health(BASE_URL):
        pytest.skip(
            f"Server not healthy at {BASE_URL}/health (start servers manually first)"
        )

    yield


def prepare_and_load_model(model_id: str) -> None:
    """Prepare topology and load model."""
    # Prepare topology
    resp = requests.post(
        f"{BASE_URL}/v1/prepare_topology",
        json={"model": model_id},
        timeout=MODEL_LOAD_TIMEOUT,
    )
    resp.raise_for_status()

    # Load model
    resp = requests.post(
        f"{BASE_URL}/v1/load_model",
        json={"model": model_id},
        timeout=MODEL_LOAD_TIMEOUT,
    )
    resp.raise_for_status()


def unload_model() -> None:
    """Unload the current model.

    Logs a warning if unloading fails, as this is a best-effort cleanup.
    """
    try:
        resp = requests.post(f"{BASE_URL}/v1/unload_model", timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to unload model (best effort): {e}")


@pytest.mark.integration
@pytest.mark.parametrize(
    "schema,prompt",
    [
        (
            {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["answer"],
            },
            "Give me a simple response with a count",
        ),
        (
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
            },
            "Create a profile for a person",
        ),
        (
            {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                "required": ["items"],
            },
            "List three fruits",
        ),
    ],
)
def test_structured_outputs_end_to_end(
    servers, schema: dict[str, Any], prompt: str
) -> None:
    """Test that structured outputs produce valid JSON conforming to schema."""
    model_id = STRUCTURED_OUTPUTS_MODEL["id"]

    try:
        # Prepare and load the model
        prepare_and_load_model(model_id)

        # Run the actual test
        _test_structured_outputs_core(schema, prompt, model_id)

    finally:
        # Cleanup: unload model
        unload_model()


def _test_structured_outputs_core(
    schema: dict[str, Any], prompt: str, model_id: str
) -> None:
    """Core test logic for structured outputs."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "structured_outputs": {"json_schema": schema},
        "max_tokens": 500,
        "temperature": 0.1,
    }

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions", json=payload, timeout=INFERENCE_TIMEOUT
    )
    assert response.status_code == 200, f"Request failed: {response.text}"

    result = response.json()
    assert "choices" in result, f"Response missing 'choices': {result}"
    assert len(result["choices"]) > 0, f"No choices in response: {result}"

    choice = result["choices"][0]
    assert "message" in choice, f"Choice missing 'message': {choice}"
    content = choice["message"].get("content", "")

    # Parse JSON - structured outputs should produce clean JSON (no end tokens)
    parsed = json.loads(content)
    assert isinstance(parsed, dict), f"Response is not a JSON object: {content}"

    # Verify it matches the schema requirements
    for required_field in schema.get("required", []):
        assert required_field in parsed, (
            f"Required field '{required_field}' missing from response"
        )

    # Basic type checking for properties
    properties = schema.get("properties", {})
    for field_name, field_schema in properties.items():
        if field_name in parsed:
            field_type = field_schema.get("type")
            if field_type == "string":
                assert isinstance(parsed[field_name], str), (
                    f"Field '{field_name}' should be string"
                )
            elif field_type == "integer":
                assert isinstance(parsed[field_name], int), (
                    f"Field '{field_name}' should be integer"
                )
            elif field_type == "array":
                assert isinstance(parsed[field_name], list), (
                    f"Field '{field_name}' should be array"
                )


@pytest.mark.integration
@pytest.mark.parametrize(
    "schema,prompt",
    [
        (
            {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["answer"],
            },
            "Give me a simple response with a count",
        ),
        (
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
            },
            "Create a profile for a person",
        ),
        (
            {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                "required": ["items"],
            },
            "List three fruits",
        ),
    ],
)
def test_openai_response_format_end_to_end(
    servers, schema: dict[str, Any], prompt: str
) -> None:
    """Test that OpenAI response_format produces valid JSON conforming to schema."""
    model_id = STRUCTURED_OUTPUTS_MODEL["id"]

    try:
        # Prepare and load the model
        prepare_and_load_model(model_id)

        # Run the actual test
        _test_openai_response_format_core(schema, prompt, model_id)

    finally:
        # Cleanup: unload model
        unload_model()


def _test_openai_response_format_core(
    schema: dict[str, Any], prompt: str, model_id: str
) -> None:
    """Core test logic for OpenAI response_format."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "test_schema", "schema": schema},
        },
        "max_tokens": 500,
        "temperature": 0.1,
    }

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions", json=payload, timeout=INFERENCE_TIMEOUT
    )
    assert response.status_code == 200, f"Request failed: {response.text}"

    result = response.json()
    assert "choices" in result, f"Response missing 'choices': {result}"
    assert len(result["choices"]) > 0, f"No choices in response: {result}"

    choice = result["choices"][0]
    assert "message" in choice, f"Choice missing 'message': {choice}"
    content = choice["message"].get("content", "")

    # Parse JSON - OpenAI response_format should produce clean JSON
    parsed = json.loads(content)
    assert isinstance(parsed, dict), f"Response is not a JSON object: {content}"

    # Verify it matches the schema requirements
    for required_field in schema.get("required", []):
        assert required_field in parsed, (
            f"Required field '{required_field}' missing from response"
        )

    # Basic type checking for properties
    properties = schema.get("properties", {})
    for field_name, field_schema in properties.items():
        if field_name in parsed:
            field_type = field_schema.get("type")
            if field_type == "string":
                assert isinstance(parsed[field_name], str), (
                    f"Field '{field_name}' should be string"
                )
            elif field_type == "integer":
                assert isinstance(parsed[field_name], int), (
                    f"Field '{field_name}' should be integer"
                )
            elif field_type == "array":
                assert isinstance(parsed[field_name], list), (
                    f"Field '{field_name}' should be array"
                )
