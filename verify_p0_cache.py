#!/usr/bin/env python3
"""
Verify P0 Features (Cache-Dependent)

Tests Skip Logic by sending repeated requests with shared prefixes.
This verifies that P0-1 (Full Skip) and P0-2 (Approximate Skip) work correctly.
"""

import asyncio
import time
from openai import AsyncOpenAI

async def test_cache_skip():
    """Test cache hit and skip logic."""

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )

    # Same system prompt (will be cached)
    system_prompt = "You are a helpful AI assistant specialized in Python programming."

    # Same user prompt prefix (for high cache hit)
    user_prompt = """Explain how Python's garbage collection works in detail, covering:
1. Reference counting
2. Cyclic garbage collection
3. Generational collection
4. Performance implications"""

    print("=" * 60)
    print("P0 Features Cache Verification")
    print("=" * 60)
    print()

    # Round 1: Cold start (no cache)
    print("🔵 Round 1: Cold start (populating cache)...")
    start = time.perf_counter()

    response1 = await client.chat.completions.create(
        model="Qwen3.5-35B-A3B-6bit",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=50,
        temperature=0.0
    )

    elapsed1 = time.perf_counter() - start
    print(f"  Time: {elapsed1:.2f}s")
    print(f"  Tokens: {response1.usage.prompt_tokens} prompt + {response1.usage.completion_tokens} completion")
    print()

    # Round 2: Cache hit (should trigger Full Skip if 100% match)
    print("🟢 Round 2: Cache hit test (expecting Full Skip)...")
    await asyncio.sleep(1)  # Brief pause

    start = time.perf_counter()

    response2 = await client.chat.completions.create(
        model="Qwen3.5-35B-A3B-6bit",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}  # Exact same prompt
        ],
        max_tokens=50,
        temperature=0.0
    )

    elapsed2 = time.perf_counter() - start
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else 0

    print(f"  Time: {elapsed2:.2f}s")
    print(f"  Tokens: {response2.usage.prompt_tokens} prompt + {response2.usage.completion_tokens} completion")
    print(f"  Speedup: {speedup:.1f}x")
    print()

    # Round 3: Partial cache hit (should trigger Approximate Skip if 95%+ match)
    print("🟡 Round 3: Partial cache hit test (expecting Approximate Skip)...")

    # Slightly different prompt (95%+ overlap)
    user_prompt_variant = user_prompt + "\n\nPlease be concise."

    await asyncio.sleep(1)
    start = time.perf_counter()

    response3 = await client.chat.completions.create(
        model="Qwen3.5-35B-A3B-6bit",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_variant}
        ],
        max_tokens=50,
        temperature=0.0
    )

    elapsed3 = time.perf_counter() - start
    speedup3 = elapsed1 / elapsed3 if elapsed3 > 0 else 0

    print(f"  Time: {elapsed3:.2f}s")
    print(f"  Tokens: {response3.usage.prompt_tokens} prompt + {response3.usage.completion_tokens} completion")
    print(f"  Speedup: {speedup3:.1f}x")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cold start:         {elapsed1:.2f}s")
    print(f"Cache hit:          {elapsed2:.2f}s (#{speedup:.1f}x)")
    print(f"Partial cache hit:  {elapsed3:.2f}s ({speedup3:.1f}x)")
    print()

    if speedup >= 2.0:
        print("✅ Full Skip Logic appears to be working (2x+ speedup)")
    else:
        print("⚠️  Full Skip Logic may not be working (expected 2x+ speedup)")

    print()
    print("📝 Check omlx_server.log for skip messages:")
    print("   grep -i 'full skip\\|approximate skip' omlx_server.log")


if __name__ == "__main__":
    asyncio.run(test_cache_skip())
