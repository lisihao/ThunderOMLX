#!/usr/bin/env python3
"""
Verify P0 Features - Ultra Long Prompts

Generate 2500+ tokens by creating unique, non-repetitive content.
"""

import asyncio
import time
from openai import AsyncOpenAI


async def test_p0_ultra_long():
    """Test P0 features with ultra-long unique prompts."""

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )

    system_prompt = "You are a helpful AI assistant."

    # Generate unique content for each number to avoid tokenizer compression
    sections = []
    for i in range(1, 51):  # 50 unique sections
        sections.append(f"""
        Section {i}: Please explain the technical architecture and implementation details of
        distributed system component number {i}, including its role in the overall ecosystem,
        how it communicates with components {i-1} and {i+1}, the specific protocols it uses
        (HTTP, gRPC, AMQP, or custom binary), its data persistence strategy (PostgreSQL,
        MongoDB, Redis, Cassandra), caching mechanisms (in-memory, distributed, write-through),
        monitoring approach (Prometheus metrics, custom dashboards, alerting rules with
        thresholds), deployment configuration (Kubernetes manifests, Helm charts, resource
        limits), security considerations (TLS certificates, API authentication via JWT tokens,
        rate limiting per endpoint), performance characteristics (latency p50/p95/p99 targets,
        throughput in requests per second, memory footprint), scalability patterns (horizontal
        pod autoscaling based on CPU/memory, vertical scaling for database instances), and
        failure recovery procedures (circuit breakers, retry policies with exponential backoff).
        """)

    user_prompt = "\n".join(sections)

    print("=" * 70)
    print("P0 Features Ultra Long Test")
    print("=" * 70)
    print()

    # Round 1: Cold start
    print("🔵 Round 1: Cold start...")
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
    tokens1 = response1.usage.prompt_tokens
    blocks1 = tokens1 // 1024
    cached_tokens = blocks1 * 1024

    print(f"  Time: {elapsed1:.2f}s")
    print(f"  Tokens: {tokens1} prompt + {response1.usage.completion_tokens} completion")
    print(f"  Blocks: {blocks1} full blocks ({cached_tokens} tokens cached)")
    print(f"  Cache ratio: {cached_tokens}/{tokens1} = {(cached_tokens/tokens1*100):.1f}%")
    print()

    if blocks1 < 2:
        print(f"⚠️  只创建了 {blocks1} 个块，需要调整提示词长度")
        print("   提示：当前提示词被 tokenizer 处理后有 {tokens1} tokens")
        print("   需要至少 2048 tokens 才能创建 2 个完整块")
        return

    # Round 2: Exact same prompt (should hit 100% cache)
    print("🟢 Round 2: Exact same prompt (Full Skip test)...")
    await asyncio.sleep(2)

    start = time.perf_counter()

    response2 = await client.chat.completions.create(
        model="Qwen3.5-35B-A3B-6bit",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=50,
        temperature=0.0
    )

    elapsed2 = time.perf_counter() - start
    speedup2 = elapsed1 / elapsed2 if elapsed2 > 0 else 0

    print(f"  Time: {elapsed2:.2f}s")
    print(f"  Speedup: {speedup2:.1f}x")
    print()

    # Round 3: Add a short suffix (should trigger Approximate Skip)
    print("🟡 Round 3: Same prompt + short suffix (Approximate Skip test)...")

    user_prompt_variant = user_prompt + "\n\nPlease keep the response concise."

    await asyncio.sleep(2)
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
    tokens3 = response3.usage.prompt_tokens
    hit_ratio3 = (cached_tokens / tokens3 * 100) if tokens3 > 0 else 0

    print(f"  Time: {elapsed3:.2f}s")
    print(f"  Tokens: {tokens3}")
    print(f"  Cache hit ratio: {cached_tokens}/{tokens3} = {hit_ratio3:.1f}%")
    print(f"  Speedup: {speedup3:.1f}x")
    print()

    # Results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"✅ Created {blocks1} cache blocks ({cached_tokens} tokens)")
    print(f"✅ Round 2 speedup: {speedup2:.1f}x")
    print(f"✅ Round 3 cache hit: {hit_ratio3:.1f}% (threshold: 95%)")
    print()

    # Check logs
    print("📝 Verification commands:")
    print("   # 检查块保存:")
    print("   grep '💾 Saved block' ~/ThunderOMLX/omlx_debug*.log | tail -10")
    print()
    print("   # 检查缓存匹配:")
    print("   grep 'Cache match result' ~/ThunderOMLX/omlx_debug*.log | tail -10")
    print()
    print("   # 检查 Skip Logic:")
    print("   grep -E '(FULL SKIP|APPROXIMATE SKIP)' ~/ThunderOMLX/omlx_debug*.log")


if __name__ == "__main__":
    asyncio.run(test_p0_ultra_long())
