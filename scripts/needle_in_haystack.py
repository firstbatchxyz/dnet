#!/usr/bin/env python3
"""
Needle in a Haystack test for Context Parallelism validation.

This test verifies that the model can attend to ALL positions in a long context,
which is essential for validating that CP is working correctly.

If CP is broken (ranks only see their chunk), the model will fail to find the needle.
This test works the best with non-thinking models such as mlx-community/Llama-3.2-3B-Instruct-4bit

Usage:
    uv run python scripts/needle_in_haystack.py --api http://localhost:8080 --context-size 4096
"""

import argparse
import random
import time
import httpx

# The "needle" - a specific fact we hide in the haystack
NEEDLE_TEMPLATE = "The secret password is: {password}"

# Filler text for the haystack (Paul Graham essays style)
HAYSTACK_CHUNKS = [
    "The most important thing in a startup is to launch quickly. "
    "You can always iterate and improve later, but you need to get "
    "something out there to learn from real users. ",
    "Good ideas look like bad ideas at first. If they looked obviously "
    "good, someone else would already be doing them. The trick is to "
    "recognize the good ideas that look bad. ",
    "Startups are about growth. A startup is a company designed to grow "
    "fast. Being newly founded does not in itself make a company a "
    "startup. Nor is it necessary for a startup to work on technology. ",
    "The way to get startup ideas is not to try to think of startup "
    "ideas. It's to look for problems, preferably problems you have "
    "yourself. The very best startup ideas tend to have three things "
    "in common: they're something the founders themselves want. ",
    "Work on hard problems. If you're working on something that seems "
    "really hard, you're probably working on something that matters. "
    "Easy problems have already been solved. ",
]


def generate_password() -> str:
    """Generate a random memorable password."""
    words = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "gamma",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
    ]
    return f"{random.choice(words)}-{random.randint(100, 999)}-{random.choice(words)}"


def generate_haystack(target_tokens: int, needle: str, needle_position: float) -> str:
    """
    Generate a haystack of approximately target_tokens with needle at specified position.

    Args:
        target_tokens: Approximate number of tokens for the haystack
        needle: The needle text to hide
        needle_position: Where to place needle (0.0 = start, 0.5 = middle, 1.0 = end)

    Returns:
        Full haystack text with needle inserted
    """
    # Rough estimate: 4 chars per token
    target_chars = target_tokens * 4

    # Build haystack chunks
    haystack_parts = []
    current_chars = 0

    while current_chars < target_chars:
        chunk = random.choice(HAYSTACK_CHUNKS)
        haystack_parts.append(chunk)
        current_chars += len(chunk)

    # Determine needle insertion point
    needle_idx = int(len(haystack_parts) * needle_position)
    needle_idx = max(1, min(needle_idx, len(haystack_parts) - 1))  # Avoid edges

    # Insert needle
    haystack_parts.insert(needle_idx, f"\n\n{needle}\n\n")

    return "".join(haystack_parts)


def run_needle_test(
    api_url: str,
    context_size: int,
    needle_position: float,
    timeout: float = 120.0,
    model: str = "default",
) -> dict:
    """
    Run a single needle in haystack test.

    Returns:
        dict with test results including success, response, latency
    """
    # Generate test case
    password = generate_password()
    needle = NEEDLE_TEMPLATE.format(password=password)
    haystack = generate_haystack(context_size, needle, needle_position)

    # Build prompt
    prompt = f"""Read the following document carefully. At some point, there is a secret password mentioned.

<document>
{haystack}
</document>

What is the secret password mentioned in the document above? Reply with ONLY the password, nothing else."""

    # Estimate actual token count
    approx_tokens = len(prompt) // 4

    print(f"\n{'=' * 60}")
    print("Needle in Haystack Test")
    print(f"{'=' * 60}")
    print(f"Target context: ~{context_size} tokens")
    print(f"Actual prompt: ~{approx_tokens} tokens")
    print(f"Needle position: {needle_position:.0%}")
    print(f"Expected password: {password}")
    print(f"{'=' * 60}")

    # Make API request
    start_time = time.time()

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{api_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,  # Qwen3 uses thinking mode, needs more tokens
                    "temperature": 0.0,  # Deterministic
                },
            )
            response.raise_for_status()
            result = response.json()
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency_s": time.time() - start_time,
            "expected": password,
            "actual": None,
        }

    latency = time.time() - start_time

    # Extract response
    try:
        actual_response = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        actual_response = str(result)

    # Check if password is in response
    success = password.lower() in actual_response.lower()

    print(f"Response: {actual_response}")
    print(f"Latency: {latency:.2f}s")
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")

    return {
        "success": success,
        "expected": password,
        "actual": actual_response,
        "latency_s": latency,
        "context_tokens": approx_tokens,
        "needle_position": needle_position,
    }


def run_full_test_suite(
    api_url: str,
    context_sizes: list[int],
    timeout: float,
    model: str = "default",
) -> None:
    """Run full test suite across context sizes and needle positions."""
    positions = [0.1, 0.25, 0.5, 0.75, 0.9]  # Test needle at different depths

    results = []

    for ctx_size in context_sizes:
        for pos in positions:
            result = run_needle_test(api_url, ctx_size, pos, timeout, model=model)
            result["target_context"] = ctx_size
            results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"Passed: {passed}/{total}")

    # Group by context size
    by_size: dict[int, list[dict]] = {}
    for r in results:
        size = r.get("target_context", 0)
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(r)

    for size in sorted(by_size.keys()):
        size_results = by_size[size]
        size_passed = sum(1 for r in size_results if r["success"])
        avg_latency = sum(r["latency_s"] for r in size_results) / len(size_results)
        print(
            f"  {size:>6} tokens: {size_passed}/{len(size_results)} passed, avg {avg_latency:.1f}s"
        )

    # Overall verdict
    if passed == total:
        print("\n✓ ALL TESTS PASSED - CP is working correctly!")
    elif passed > total // 2:
        print("\n⚠ PARTIAL PASS - Some positions may have issues")
    else:
        print("\n✗ TESTS FAILED - CP may not be attending to full context")


def main():
    parser = argparse.ArgumentParser(
        description="Needle in a Haystack test for CP validation"
    )
    parser.add_argument("--api", default="http://localhost:8080", help="API server URL")
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="Single context size to test (default: run full suite)",
    )
    parser.add_argument(
        "--sizes",
        default="512,1024,2048,4096,8192,16384,32768",
        help="Comma-separated context sizes for full suite",
    )
    parser.add_argument(
        "--position",
        type=float,
        default=0.5,
        help="Needle position (0.0-1.0) for single test",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Request timeout in seconds"
    )

    parser.add_argument(
        "--model",
        default="default",
        help="Model name to use for requests",
    )

    args = parser.parse_args()

    if args.context_size:
        # Single test
        run_needle_test(
            args.api, args.context_size, args.position, args.timeout, model=args.model
        )
    else:
        # Full suite
        sizes = [int(s.strip()) for s in args.sizes.split(",")]
        run_full_test_suite(args.api, sizes, args.timeout, model=args.model)


if __name__ == "__main__":
    main()
