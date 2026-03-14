#!/usr/bin/env python3
"""
Verify P0 Features - Final Test with 2100+ tokens

Creates prompts long enough to generate 2+ full cache blocks (2048+ tokens cached)
This ensures cache hit ratio > 95% for APPROXIMATE SKIP testing.
"""

import asyncio
import time
from openai import AsyncOpenAI


async def test_p0_features():
    """Test all P0 features with proper block sizes."""

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )

    # Generate a very long prompt by repeating detailed content
    # Target: 2100+ tokens to create 2 full blocks (2048 tokens cached)

    system_prompt = """You are an expert software architect with 15+ years of experience in:
    - Distributed systems design and implementation
    - High-performance computing and optimization
    - Cloud-native architecture patterns
    - Database systems and query optimization
    - Machine learning infrastructure
    - DevOps and site reliability engineering

    Provide detailed, production-ready solutions with code examples."""

    # Create extremely long content by repeating technical specifications
    tech_section = """

    === SECTION: DISTRIBUTED CACHING ARCHITECTURE ===

    Design a comprehensive distributed caching system with the following requirements:

    1. CACHE TOPOLOGY:
       - Multi-tier caching (L1: local memory, L2: Redis cluster, L3: SSD)
       - Geographic distribution across 5 AWS regions
       - Automatic failover and replication
       - Consistent hashing for data distribution
       - Read-through and write-through patterns

    2. PERFORMANCE REQUIREMENTS:
       - Sub-millisecond p99 latency for L1 cache
       - <5ms p99 latency for L2 cache (Redis)
       - <50ms p99 latency for L3 cache (SSD)
       - Support 1M+ requests per second per region
       - 99.99% availability SLA

    3. DATA MANAGEMENT:
       - Configurable TTL per cache entry
       - LRU eviction with size-aware policies
       - Probabilistic early expiration (beta distribution)
       - Cache stampede prevention via request coalescing
       - Versioning for safe rollbacks

    4. CONSISTENCY MODEL:
       - Strong consistency within region
       - Eventual consistency across regions
       - Conflict resolution using vector clocks
       - Read-your-writes guarantee
       - Monotonic reads guarantee

    5. MONITORING AND OBSERVABILITY:
       - Real-time metrics (hit rate, miss rate, latency percentiles)
       - Distributed tracing with OpenTelemetry
       - Alerting on cache degradation
       - Capacity planning dashboards
       - Cost attribution per service
    """

    # Repeat the section 4 times to reach 2100+ tokens
    user_prompt = f"""Please provide a detailed technical design document covering:

    {tech_section}
    {tech_section}
    {tech_section}
    {tech_section}

    For the distributed caching architecture described above, provide:

    - Complete system design with component diagrams (described in text)
    - Redis Cluster configuration (sentinel vs cluster mode)
    - Network topology and data flow
    - Failure scenarios and recovery procedures
    - Capacity planning calculations
    - Sample configuration files for Redis, Nginx, and application code
    - Monitoring queries and alerting rules
    - Cost breakdown for AWS infrastructure
    - Migration strategy from current system
    - Performance benchmarking methodology

    Please be extremely detailed and production-ready. Include specific:
    - AWS instance types and sizes
    - Redis configuration parameters (maxmemory, eviction policies)
    - Network security groups and VPC configuration
    - IAM roles and permissions
    - CloudWatch metrics and alarms
    - Terraform/CloudFormation templates
    - Load testing scripts and results
    - Disaster recovery runbooks
    """

    print("=" * 70)
    print("P0 Features Final Verification (2100+ tokens)")
    print("=" * 70)
    print()

    # Round 1: Cold start
    print("🔵 Round 1: Cold start (creating 2+ cache blocks)...")
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

    print(f"  Time: {elapsed1:.2f}s")
    print(f"  Tokens: {tokens1} prompt + {response1.usage.completion_tokens} completion")
    print(f"  Blocks: {blocks1} full blocks created ({blocks1 * 1024} tokens cached)")
    print(f"  Remainder: {tokens1 % 1024} tokens (not cached)")
    print()

    # Round 2: Full cache hit
    print("🟢 Round 2: Cache hit test (expecting FULL SKIP with 100% hit)...")
    await asyncio.sleep(2)

    start = time.perf_counter()

    response2 = await client.chat.completions.create(
        model="Qwen3.5-35B-A3B-6bit",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}  # Exact same
        ],
        max_tokens=50,
        temperature=0.0
    )

    elapsed2 = time.perf_counter() - start
    speedup2 = elapsed1 / elapsed2 if elapsed2 > 0 else 0

    print(f"  Time: {elapsed2:.2f}s")
    print(f"  Tokens: {response2.usage.prompt_tokens} prompt + {response2.usage.completion_tokens} completion")
    print(f"  Speedup: {speedup2:.1f}x")
    if blocks1 * 1024 == response2.usage.prompt_tokens:
        print(f"  Expected: FULL SKIP (100% cache hit)")
    else:
        expected_ratio = (blocks1 * 1024) / response2.usage.prompt_tokens * 100
        print(f"  Expected: Cache hit ratio = {expected_ratio:.1f}%")
    print()

    # Round 3: Partial cache hit (add a few words)
    print("🟡 Round 3: Partial cache hit (expecting APPROXIMATE SKIP >95%)...")

    user_prompt_variant = user_prompt + "\n\nPlease prioritize cost optimization."

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
    expected_hit_ratio = (blocks1 * 1024) / tokens3 * 100

    print(f"  Time: {elapsed3:.2f}s")
    print(f"  Tokens: {tokens3} prompt + {response3.usage.completion_tokens} completion")
    print(f"  Speedup: {speedup3:.1f}x")
    print(f"  Expected cache hit: {blocks1 * 1024}/{tokens3} = {expected_hit_ratio:.1f}%")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Cold start:         {elapsed1:.2f}s  ({tokens1} tokens → {blocks1} blocks)")
    print(f"Cache hit:          {elapsed2:.2f}s  ({speedup2:.1f}x speedup)")
    print(f"Partial cache hit:  {elapsed3:.2f}s  ({speedup3:.1f}x speedup)")
    print()

    # Verification
    success = True
    if blocks1 < 2:
        print(f"⚠️  Warning: Only {blocks1} blocks created (need 2+ for proper testing)")
        success = False
    else:
        print(f"✅ Created {blocks1} full cache blocks")

    if speedup2 >= 2.0:
        print(f"✅ Round 2 speedup: {speedup2:.1f}x (cache working)")
    else:
        print(f"⚠️  Round 2 speedup: {speedup2:.1f}x (expected ≥2x)")
        success = False

    if expected_hit_ratio >= 95.0:
        print(f"✅ Round 3 cache hit: {expected_hit_ratio:.1f}% (≥95%, should trigger APPROXIMATE SKIP)")
    else:
        print(f"⚠️  Round 3 cache hit: {expected_hit_ratio:.1f}% (<95%, won't trigger APPROXIMATE SKIP)")
        success = False

    print()
    print("=" * 70)
    print("VERIFICATION STEPS")
    print("=" * 70)
    print()
    print("1️⃣ Check if blocks were saved to SSD:")
    print("   grep '💾 Saved block' ~/ThunderOMLX/omlx_debug*.log | tail -5")
    print()
    print("2️⃣ Check cache match results:")
    print("   grep 'Cache match result' ~/ThunderOMLX/omlx_debug*.log | tail -5")
    print()
    print("3️⃣ Check for FULL SKIP or APPROXIMATE SKIP logs:")
    print("   grep -i 'full skip\\|approximate skip' ~/ThunderOMLX/omlx_debug*.log")
    print()
    print("4️⃣ Check SSD cache files:")
    print("   find ~/.cache/omlx_cache -type f -name '*.safetensors*'")
    print("   du -sh ~/.cache/omlx_cache")
    print()

    if success:
        print("🎉 All checks passed! P0 features should be working.")
    else:
        print("⚠️  Some checks failed. Review the verification steps above.")


if __name__ == "__main__":
    asyncio.run(test_p0_features())
