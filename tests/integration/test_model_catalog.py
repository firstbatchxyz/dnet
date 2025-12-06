"""Integration tests for model catalog.

These tests validate that small models work end-to-end on a single machine.
Tests are parameterized directly from catalog.py and run serially.

Usage (with servers running):
    uv run pytest tests/integration/test_model_catalog.py -v -x

Usage (standalone - starts servers automatically):
    uv run pytest tests/integration/test_model_catalog.py -v -x --start-servers
"""

import os
import signal
import subprocess
import sys
import time
from typing import Generator

import pytest
import requests

from dnet.api.catalog import get_ci_test_models

# Server configuration
API_HTTP_PORT = 8080
API_GRPC_PORT = 58080
SHARD_HTTP_PORT = 8081
SHARD_GRPC_PORT = 58081
BASE_URL = f"http://localhost:{API_HTTP_PORT}"

# Timeouts
HEALTH_CHECK_TIMEOUT = 60  # seconds to wait for servers to start
MODEL_LOAD_TIMEOUT = 300  # seconds to wait for model loading
INFERENCE_TIMEOUT = 120  # seconds for inference


def pytest_addoption(parser):
    """Add --start-servers option to pytest."""
    parser.addoption(
        "--start-servers",
        action="store_true",
        default=False,
        help="Start dnet-api and dnet-shard servers automatically",
    )


@pytest.fixture(scope="module")
def start_servers_flag(request):
    """Get the --start-servers flag value."""
    return request.config.getoption("--start-servers")


def wait_for_health(url: str, timeout: float = HEALTH_CHECK_TIMEOUT) -> bool:
    """Wait for server health endpoint to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def servers(start_servers_flag) -> Generator[None, None, None]:
    """Start API and shard servers if --start-servers flag is set.

    Otherwise, assumes servers are already running and just checks health.
    """
    procs: list[subprocess.Popen] = []

    if start_servers_flag:
        # Start shard first
        shard_cmd = [
            sys.executable,
            "-m",
            "cli.shard",
            "--http-port",
            str(SHARD_HTTP_PORT),
            "--grpc-port",
            str(SHARD_GRPC_PORT),
        ]
        shard_proc = subprocess.Popen(
            shard_cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            env={**os.environ, "PYTHONPATH": "src"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(shard_proc)
        time.sleep(2)  # Give shard time to start

        # Start API
        api_cmd = [
            sys.executable,
            "-m",
            "cli.api",
            "--http-port",
            str(API_HTTP_PORT),
            "--grpc-port",
            str(API_GRPC_PORT),
        ]
        api_proc = subprocess.Popen(
            api_cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            env={**os.environ, "PYTHONPATH": "src"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(api_proc)

    # Wait for API health
    if not wait_for_health(BASE_URL):
        # Cleanup if we started servers
        for p in procs:
            p.terminate()
            p.wait(timeout=5)
        pytest.skip(f"Server not healthy at {BASE_URL}/health")

    yield

    # Cleanup: terminate servers if we started them
    for p in procs:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


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
    """Unload the current model."""
    try:
        requests.post(f"{BASE_URL}/v1/unload_model", timeout=30)
    except requests.RequestException:
        pass  # Best effort


# Get CI-testable models from catalog
CI_TEST_MODELS = get_ci_test_models()


@pytest.mark.integration
@pytest.mark.parametrize(
    "model",
    CI_TEST_MODELS,
    ids=[m["alias"] for m in CI_TEST_MODELS],
)
def test_model_inference(servers, model: dict) -> None:
    """Test that a model can load and produce meaningful inference output.

    This test:
    1. Prepares the topology for the model
    2. Loads the model onto the shard(s)
    3. Sends a simple prompt
    4. Validates the response is non-empty and meaningful
    """
    model_id = model["id"]

    try:
        # Prepare and load
        prepare_and_load_model(model_id)

        # Send inference request
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {"role": "user", "content": "What is 2+2? Reply briefly."}
                ],
                "max_tokens": 50,
                "temperature": 0.1,
            },
            timeout=INFERENCE_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()

        # Validate response structure
        assert "choices" in result, f"Response missing 'choices': {result}"
        assert len(result["choices"]) > 0, f"No choices in response: {result}"

        # Validate content
        choice = result["choices"][0]
        assert "message" in choice, f"Choice missing 'message': {choice}"
        content = choice["message"].get("content", "")
        assert len(content) > 0, f"Empty response content for model {model_id}"

        # Basic sanity check - response should contain something meaningful
        # (not just whitespace or error messages)
        assert content.strip(), f"Response is only whitespace: {repr(content)}"

        print(f"\n✓ Model {model['alias']} responded: {content[:100]}...")

    finally:
        # Always try to unload to prepare for next test
        unload_model()
