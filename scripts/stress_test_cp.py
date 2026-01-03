#!/usr/bin/env python3
"""
Stress test for Context Parallelism via the chat completions endpoint.

Sends requests with varying prompt lengths to test CP's ability to handle
long contexts distributed across shards.

Usage:
    uv run scripts/stress_test_cp.py
    uv run scripts/stress_test_cp.py --api http://10.0.0.1:8080 --max-tokens 1000
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add project root to sys.path to allow imports from scripts package
sys.path.append(str(Path(__file__).parent.parent))

import requests

from dnet.api.models import ChatMessage, ChatRequestModel, ChatResponseModel

from scripts.cp_utils import (
    get_shards,
    get_topology,
    get_recommended_test_sizes,
    is_cp_enabled,
)


@dataclass
class TestResult:
    """Result of a single stress test run."""

    context_length: int
    prompt_chars: int
    success: bool
    total_time_s: float
    time_to_first_token_s: Optional[float] = None
    num_chunks: Optional[int] = None
    response: Optional[ChatResponseModel] = None
    error: Optional[str] = None
    stream: bool = False


def generate_long_prompt(target_tokens: int) -> str:
    """Generate a prompt of approximately target_tokens length.

    Uses repetitive text to reach target length. Rough estimate: 1 token ≈ 4 chars.
    """
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
    )
    target_chars = target_tokens * 4
    repetitions = max(1, target_chars // len(base_text))
    return base_text * repetitions


def run_chat_request(
    api_url: str,
    prompt: str,
    context_length: int,
    max_tokens: int = 50,
    stream: bool = False,
    timeout: int = 3600,  # 60 min for long contexts (64K+ tokens)
) -> TestResult:
    """Send a chat completion request and return typed TestResult."""
    request = ChatRequestModel(
        model="default",
        messages=[
            ChatMessage(role="user", content=prompt),
        ],
        max_tokens=max_tokens,
        stream=stream,
        temperature=0.7,
    )

    prompt_chars = len(prompt)
    start_time = time.time()

    if stream:
        try:
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                json=request.model_dump(),
                stream=True,
                timeout=timeout,
            )
            if not response.ok:
                return TestResult(
                    context_length=context_length,
                    prompt_chars=prompt_chars,
                    success=False,
                    total_time_s=time.time() - start_time,
                    error=f"{response.status_code} {response.reason}: {response.text}",
                    stream=True,
                )

            chunks = []
            first_token_time: Optional[float] = None
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        if first_token_time is None:
                            first_token_time = time.time()
                        chunks.append(decoded[6:])

            end_time = time.time()
            return TestResult(
                context_length=context_length,
                prompt_chars=prompt_chars,
                success=True,
                total_time_s=end_time - start_time,
                time_to_first_token_s=(first_token_time - start_time)
                if first_token_time
                else None,
                num_chunks=len(chunks),
                stream=True,
            )
        except requests.RequestException as e:
            return TestResult(
                context_length=context_length,
                prompt_chars=prompt_chars,
                success=False,
                total_time_s=time.time() - start_time,
                error=str(e),
                stream=True,
            )

    else:
        try:
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                json=request.model_dump(),
                timeout=timeout,
            )
            if not response.ok:
                return TestResult(
                    context_length=context_length,
                    prompt_chars=prompt_chars,
                    success=False,
                    total_time_s=time.time() - start_time,
                    error=f"{response.status_code} {response.reason}: {response.text}",
                    stream=False,
                )

            end_time = time.time()
            chat_response = ChatResponseModel.model_validate(response.json())
            return TestResult(
                context_length=context_length,
                prompt_chars=prompt_chars,
                success=True,
                total_time_s=end_time - start_time,
                response=chat_response,
                stream=False,
            )
        except requests.RequestException as e:
            return TestResult(
                context_length=context_length,
                prompt_chars=prompt_chars,
                success=False,
                total_time_s=time.time() - start_time,
                error=str(e),
                stream=False,
            )


def run_stress_test(
    api_url: str,
    context_lengths: list[int],
    max_tokens: int,
    stream: bool,
    verbose: bool,
) -> list[TestResult]:
    """Run stress tests with varying context lengths."""
    results: list[TestResult] = []

    for ctx_len in context_lengths:
        print(f"\n[Test] Context length: ~{ctx_len:,} tokens")
        prompt = generate_long_prompt(ctx_len)
        actual_chars = len(prompt)
        print(f"       Prompt: {actual_chars:,} chars (~{actual_chars // 4:,} tokens)")

        try:
            result = run_chat_request(
                api_url=api_url,
                prompt=prompt,
                context_length=ctx_len,
                max_tokens=max_tokens,
                stream=stream,
            )
            results.append(result)

            if result.success:
                print(f"       ✓ Success in {result.total_time_s:.2f}s")
                if stream and result.time_to_first_token_s:
                    print(
                        f"       Time to first token: {result.time_to_first_token_s:.2f}s"
                    )
                if verbose and not stream and result.response:
                    resp = result.response
                    if resp.choices:
                        msg = resp.choices[0].message
                        content = msg.content if msg else ""
                        print(f"       Response: {content[:100]}...")
                    if resp.usage:
                        print(
                            f"       Tokens: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}"
                        )
        except requests.RequestException as e:
            print(f"       ✗ Failed: {e}")
            results.append(
                TestResult(
                    context_length=ctx_len,
                    prompt_chars=len(prompt),
                    success=False,
                    total_time_s=0.0,
                    error=str(e),
                )
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stress test Context Parallelism via chat endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api",
        type=str,
        default="http://localhost:8080",
        help="API server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming responses",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show response content",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with small context lengths only",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated context sizes to test (default: auto based on shard count)",
    )
    args = parser.parse_args()

    api_url = args.api.rstrip("/")

    print("=" * 60)
    print("Context Parallelism Stress Test")
    print("=" * 60)
    print(f"API:        {api_url}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Streaming:  {args.stream}")

    # Get shard count for test size recommendations
    print("\n[Check] Detecting shards...")
    try:
        shards = get_shards(api_url)
        num_shards = len(shards)
        print(f"        Found {num_shards} shard(s):")
        for s in shards:
            print(f"          - {s.instance} ({s.local_ip}:{s.server_port})")
    except requests.RequestException as e:
        print(f"        Warning: Could not fetch shards: {e}")
        num_shards = 1

    # Verify model is loaded
    print("\n[Check] Verifying model is loaded...")
    topo = get_topology(api_url)
    if topo:
        print(f"        Model: {topo.get('model', 'unknown')}")
    else:
        print("        Warning: Could not fetch topology")

    # Check if CP is enabled
    print("\n[Check] Checking CP settings...")
    cp_enabled = is_cp_enabled(api_url)
    if cp_enabled:
        print("        ✓ Context Parallelism is ENABLED")
    else:
        print("        ⚠ Context Parallelism is DISABLED (DNET_CP_ENABLED=false)")
        print("          Tests will run in single-device mode")

    # Determine test context lengths
    if args.sizes:
        context_lengths = [int(s.strip()) for s in args.sizes.split(",")]
    elif args.quick:
        context_lengths = [100, 500, 1000]
    else:
        context_lengths = get_recommended_test_sizes(num_shards)

    print(f"\nTest sizes: {context_lengths}")
    if num_shards > 1:
        print(
            f"(Recommended for {num_shards} shards - includes sizes that benefit from CP)"
        )

    # Run tests
    results = run_stress_test(
        api_url=api_url,
        context_lengths=context_lengths,
        max_tokens=args.max_tokens,
        stream=args.stream,
        verbose=args.verbose,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"Tests passed: {len(successful)}/{len(results)}")
    print(f"Shards used:  {num_shards}")

    if successful:
        times = [r.total_time_s for r in successful]
        print(f"Avg time:     {sum(times) / len(times):.2f}s")
        print(f"Max time:     {max(times):.2f}s")

        print("\nDetails:")
        print(f"{'Context':<10} {'Time':<10} {'TTFT':<10} {'Tokens/s':<10}")
        print("-" * 45)
        for r in successful:
            tokens_per_sec = ""
            if r.response and r.response.usage:
                total_tokens = (
                    r.response.usage.prompt_tokens + r.response.usage.completion_tokens
                )
                tps = total_tokens / r.total_time_s
                tokens_per_sec = f"{tps:.1f}"

            ttft = f"{r.time_to_first_token_s:.2f}s" if r.time_to_first_token_s else "-"
            print(
                f"{r.context_length:<10} {r.total_time_s:<10.2f} {ttft:<10} {tokens_per_sec:<10}"
            )

    if failed:
        print("\nFailed tests:")
        for r in failed:
            err = r.error or "unknown error"
            print(f"  - {r.context_length:,} tokens: {err}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
