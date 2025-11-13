"""Tests for OpenAI compatibility.
The test is written so that a Python OpenAI client is used to make requests,
and the responses are verified to be compatible with OpenAI's API.
Similar usage can be seen on [Ollama](https://github.com/ollama/ollama/blob/main/docs/openai.md).
"""

from openai import OpenAI
import pytest
import requests


MODEL = "Qwen/Qwen3-4B-MLX-4bit"
BASE_URL = "http://localhost:8080"


@pytest.fixture(scope="module")
def client():
    """Create OpenAI client and check server health."""
    # Check if server is responding
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        response.raise_for_status()
    except (requests.RequestException, requests.ConnectionError):
        pytest.skip("Server is not responding at /health endpoint")

    return OpenAI(
        base_url=f"{BASE_URL}/v1/",  # dnet API url, with `v1`
        api_key="dria",  # ignored
    )


def test_openai_completions(client):
    """https://platform.openai.com/docs/api-reference/completions/create"""
    completion = client.completions.create(
        model=MODEL,
        prompt="Say this is a test",
        max_tokens=5,
    )

    print(completion)


def test_openai_chat_completions(client):
    """https://platform.openai.com/docs/api-reference/chat/create"""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model=MODEL,
    )

    print(chat_completion)


def test_openai_chat_completions_streaming(client):
    """https://platform.openai.com/docs/api-reference/chat/create (with streaming: true)"""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model=MODEL,
        stream=True,
    )

    print(chat_completion)


@pytest.mark.skip(reason="models endpoint not yet added")
def test_openai_models(client):
    """https://platform.openai.com/docs/api-reference/models/list"""
    list_completion = client.models.list()

    # should have MODEL in the list
    print(list_completion)


@pytest.mark.skip(reason="embeddings endpoint not yet added")
def test_openai_embeddings(client):
    """https://platform.openai.com/docs/api-reference/embeddings/create"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input="The food was delicious and the service was excellent.",
    )
    print(response)
