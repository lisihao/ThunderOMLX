#!/usr/bin/env python3
"""
Verify P0 Features with Long Prompts (>1024 tokens)

Tests cache-dependent features with prompts long enough to create full blocks.
Block size = 1024 tokens, so we need prompts with 1100+ tokens.
"""

import asyncio
import time
from openai import AsyncOpenAI


async def test_cache_skip_long():
    """Test cache hit with long prompts (>1024 tokens)."""

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )

    # Create a long system prompt (200+ tokens)
    system_prompt = """You are an expert AI assistant specializing in computer science,
    software engineering, and system architecture. You have deep knowledge of:
    - Programming languages: Python, JavaScript, TypeScript, Rust, Go, C++, Java
    - Web technologies: React, Vue, Angular, Node.js, Express, FastAPI
    - Databases: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch
    - Cloud platforms: AWS, GCP, Azure, Kubernetes, Docker
    - Machine Learning: PyTorch, TensorFlow, scikit-learn, Transformers
    - System Design: Microservices, Event-driven architecture, CQRS, DDD
    - DevOps: CI/CD, GitOps, Infrastructure as Code, Monitoring

    Your responses should be detailed, technically accurate, and include practical examples.
    Always explain complex concepts clearly and provide code examples when relevant.
    """

    # Create a long user prompt (1200+ tokens)
    # Repeat detailed technical content to reach token count
    base_content = """Please provide a comprehensive explanation of how modern web application
    caching strategies work, covering the following aspects in detail:"""

    # Repeat content multiple times to ensure >1024 tokens
    user_prompt = base_content + "\n\n" + """
    [PART 1 - FOUNDATIONS]
    Please provide a comprehensive explanation of how modern web application
    caching strategies work, covering the following aspects in detail:

    1. CLIENT-SIDE CACHING:
       - Browser cache mechanisms (HTTP caching headers: Cache-Control, ETag, Last-Modified)
       - Service Workers and Cache API for offline-first progressive web apps
       - LocalStorage and SessionStorage best practices and limitations
       - IndexedDB for structured data caching
       - Memory caching in JavaScript applications

    2. CDN CACHING:
       - Content Delivery Network architecture and edge caching
       - Cache invalidation strategies (purge, ban, surrogate keys)
       - Geographic distribution and origin shield
       - Cache hit ratio optimization techniques
       - Dynamic content caching at the edge

    3. APPLICATION-LEVEL CACHING:
       - In-memory caching with Redis and Memcached
       - Cache-aside (lazy loading) pattern implementation
       - Write-through and write-behind caching patterns
       - Cache warming strategies for high-traffic scenarios
       - Distributed caching in microservices architectures

    4. DATABASE QUERY CACHING:
       - Query result caching in PostgreSQL and MySQL
       - ORM-level caching in Django, SQLAlchemy, TypeORM
       - Materialized views for complex aggregations
       - Read replicas and query routing strategies

    5. API RESPONSE CACHING:
       - HTTP caching for RESTful APIs
       - GraphQL query result caching and normalization
       - Conditional requests (304 Not Modified)
       - Versioning strategies for cache invalidation

    6. ADVANCED PATTERNS:
       - Cache stampede prevention (request coalescing)
       - Probabilistic early expiration to avoid thundering herd
       - Multi-tier caching hierarchies (L1/L2/L3)
       - Cache consistency in distributed systems
       - Eventual consistency trade-offs

    7. PERFORMANCE OPTIMIZATION:
       - Cache key design and namespace strategies
       - TTL (Time To Live) tuning based on data volatility
       - Cache size limits and eviction policies (LRU, LFU, FIFO)
       - Monitoring cache hit rates and performance metrics
       - A/B testing different caching strategies

    8. COMMON PITFALLS:
       - Cache invalidation complexity ("There are only two hard things...")
       - Stale data issues and mitigation strategies
       - Memory pressure and cache size management
       - Network latency in distributed caching
       - Serialization overhead for complex objects

    For each section, please provide:
    - Detailed technical explanation
    - Code examples where applicable
    - Best practices and anti-patterns
    - Real-world use cases and scenarios
    - Performance implications and benchmarks

    This is for a production system handling millions of requests per day, so accuracy
    and practical applicability are critical. Please be thorough and comprehensive in
    your explanation, as this will serve as documentation for our engineering team.

    [PART 2 - ADDITIONAL CONTEXT]
    Additionally, please consider these real-world scenarios and challenges:

    - High-concurrency environments with 100,000+ requests per second
    - Global distribution across multiple data centers and regions
    - Mobile-first applications with varying network conditions
    - Real-time data updates and cache invalidation requirements
    - Cost optimization for cloud infrastructure (AWS, GCP, Azure)
    - Security considerations (cache poisoning, timing attacks)
    - Compliance requirements (GDPR, data residency)
    - Monitoring and observability best practices
    - Disaster recovery and failover scenarios
    - Performance SLAs (p50, p95, p99 latency targets)

    Please structure your response with clear headings, code examples in multiple
    languages (Python, JavaScript, Go), and architectural diagrams described in text.
    Include specific configuration examples for popular technologies like Redis,
    Memcached, Varnish, and CloudFlare. Discuss trade-offs between different
    approaches and provide decision frameworks for choosing the right strategy.

    [PART 3 - TECHNICAL DEPTH]
    For each caching layer, explain the underlying implementation details:
    - Memory management and garbage collection impact
    - Serialization/deserialization performance
    - Network protocol overhead (Redis RESP, Memcached binary protocol)
    - Consistency models (strong, eventual, causal)
    - Conflict resolution strategies
    - Partition tolerance in distributed caches
    - Read-through and write-through cache patterns
    - Cache warming and pre-loading strategies
    - Metrics and monitoring (hit rate, miss rate, latency, memory usage)
    - Alerting thresholds and incident response procedures
    """

    print("=" * 60)
    print("P0 Features Cache Verification (Long Prompts)")
    print("=" * 60)
    print()

    # Round 1: Cold start (no cache, will create blocks)
    print("🔵 Round 1: Cold start (populating cache with >1024 tokens)...")
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
    print(f"  Expected: {response1.usage.prompt_tokens} tokens should create ~{response1.usage.prompt_tokens // 1024} full blocks")
    print()

    # Round 2: Cache hit (should trigger Full Skip if 100% match)
    print("🟢 Round 2: Cache hit test (expecting FULL SKIP)...")
    await asyncio.sleep(2)  # Brief pause to let cache settle

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

    # Round 3: Partial cache hit
    print("🟡 Round 3: Partial cache hit test (expecting APPROXIMATE SKIP)...")

    user_prompt_variant = user_prompt + "\n\nPlease prioritize practical examples over theory."

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

    print(f"  Time: {elapsed3:.2f}s")
    print(f"  Tokens: {response3.usage.prompt_tokens} prompt + {response3.usage.completion_tokens} completion")
    print(f"  Speedup: {speedup3:.1f}x")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cold start:         {elapsed1:.2f}s")
    print(f"Cache hit:          {elapsed2:.2f}s ({speedup:.1f}x)")
    print(f"Partial cache hit:  {elapsed3:.2f}s ({speedup3:.1f}x)")
    print()

    if speedup >= 2.0:
        print("✅ Full Skip Logic appears to be working (2x+ speedup)")
    else:
        print("⚠️  Full Skip Logic may not be working (expected 2x+ speedup)")

    print()
    print("📝 Check server logs for cache messages:")
    print("   grep -i 'full skip\\|approximate skip\\|saved block' omlx_debug*.log")
    print()
    print("📂 Check SSD cache files:")
    print("   find ~/.cache/omlx_cache -type f -name '*.safetensors*' | head -5")
    print("   du -sh ~/.cache/omlx_cache")


if __name__ == "__main__":
    asyncio.run(test_cache_skip_long())
