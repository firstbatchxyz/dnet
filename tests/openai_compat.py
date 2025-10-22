"""Tests for OpenAI compatibility.

The test is written so that a Python OpenAI client is used to make requests,
and the responses are verified to be compatible with OpenAI's API.

Similar usage can be seen on [Ollama](https://github.com/ollama/ollama/blob/main/docs/openai.md).
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1/",  # dnet API url, with `v1`
    api_key="dria",  # ignored
)


def test_openai_completions():
    """https://platform.openai.com/docs/api-reference/completions/create"""
    completion = client.completions.create(
        model="llama3.2",
        prompt="Say this is a test",
        max_tokens=5,
    )

    print(completion)


def test_openai_chat_completions():
    """https://platform.openai.com/docs/api-reference/chat/create"""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="llama3.2",
    )

    print(chat_completion)


def test_openai_chat_completions_streaming():
    """https://platform.openai.com/docs/api-reference/chat/create (with streaming: true)"""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="llama3.2",
        stream=True,
    )

    print(chat_completion)


def test_openai_models():
    """https://platform.openai.com/docs/api-reference/models/list"""
    list_completion = client.models.list()
    print(list_completion)


def test_openai_embeddings():
    """https://platform.openai.com/docs/api-reference/embeddings/create"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input="The food was delicious and the service was excellent.",
    )
    print(response)
