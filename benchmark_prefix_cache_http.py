#!/usr/bin/env python3
"""
ThunderOMLX Prefix Caching Benchmark (HTTP API)

正确的测试方式：通过 ThunderOMLX 的 HTTP API 测试 BlockAwarePrefixCache

与之前错误的测试对比：
- ❌ 之前：直接使用 mlx-lm 的 make_prompt_cache() → 测试了错误的系统
- ✅ 现在：通过 ThunderOMLX HTTP API → 自动使用 BlockAwarePrefixCache + ContextPilotAdapter

前提条件：
1. 启动 ThunderOMLX 服务器
   python -m omlx.server \\
     --model /Users/lisihao/models/qwen3.5-35b-mlx \\
     --port 8080 \\
     --paged-ssd-cache-dir ~/.cache/omlx/paged_ssd

2. 验证配置：
   curl -s http://localhost:8080/cache/stats | jq .
"""

import argparse
import json
import time
from typing import Dict, List

import requests


def measure_ttft(
    api_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 32,
) -> float:
    """
    测量 TTFT（Time To First Token）

    Args:
        api_url: ThunderOMLX API endpoint
        messages: Messages (system + user)
        max_tokens: Max tokens to generate

    Returns:
        TTFT in milliseconds
    """
    payload = {
        "model": "qwen3.5-35b",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }

    start = time.perf_counter()
    response = requests.post(api_url, json=payload, stream=True)

    # Read first chunk (first token)
    for line in response.iter_lines():
        if line:
            first_token_time = time.perf_counter()
            ttft = (first_token_time - start) * 1000  # ms
            return ttft

    return None


def benchmark_openclawworkload(
    api_url: str,
    system_prompt_length: int = 800,
    num_agents: int = 5,
    queries_per_agent: int = 3,
):
    """
    Benchmark ThunderOMLX Prefix Caching with OpenClaw workload.

    OpenClaw 场景特征：
    - 5 个 agent 类型 (pm-agent, researcher-agent, etc.)
    - 每个 agent 有固定的 system prompt (~800 tokens)
    - 80%+ 的请求共享相同的 system prompt

    预期效果：
    - 第一次请求：TTFT ~1000ms (full prefill)
    - 后续请求：TTFT ~200ms (cache hit) → **-80% TTFT** ⭐
    """

    print("=" * 80)
    print("🚀 ThunderOMLX Prefix Caching Benchmark (HTTP API)")
    print("=" * 80)

    print(f"\n📊 Configuration:")
    print(f"   API URL: {api_url}")
    print(f"   System Prompt Length: {system_prompt_length} tokens")
    print(f"   Num Agents: {num_agents}")
    print(f"   Queries per Agent: {queries_per_agent}")

    # Check if server is running and cache is enabled
    try:
        stats_response = requests.get(f"{api_url.replace('/v1/chat/completions', '/cache/stats')}")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"\n✅ Server cache status:")
            print(f"   Enabled: {stats.get('enabled', False)}")
            print(f"   Type: {stats.get('type', 'unknown')}")
            if not stats.get('enabled'):
                print("\n⚠️  WARNING: Cache is disabled! Prefix caching will not work.")
                print("   Make sure --paged-ssd-cache-dir is set when starting the server.")
        else:
            print(f"\n⚠️  Could not check cache status (status code: {stats_response.status_code})")
    except Exception as e:
        print(f"\n⚠️  Could not check cache status: {e}")

    # Create fixed system prompts for each agent
    agent_system_prompts = {}
    for i in range(num_agents):
        agent_name = f"agent-{i+1}"
        # Create a system prompt (~800 tokens)
        system_prompt = (
            f"You are {agent_name}, a specialized AI assistant. "
            "Your role is to provide accurate, detailed, and well-structured responses. "
            "Always be polite, professional, and clear in your communication. "
        ) * 50  # Repeat to reach ~800 tokens

        # Trim to exact length (approximate)
        # Note: In real scenario, we'd tokenize to get exact token count
        agent_system_prompts[agent_name] = system_prompt[:system_prompt_length * 4]  # ~4 chars/token

    print(f"\n✅ Created {num_agents} agents with fixed system prompts")

    # User queries
    user_queries = [
        "What is the status of the project?",
        "Can you help me with this task?",
        "Please review the recent changes.",
        "What are the next steps?",
        "Any issues to address?",
        "How can I improve this?",
        "What's your recommendation?",
        "Can you summarize this?",
        "What do you think?",
        "Please provide feedback.",
    ]

    results = {"cold": [], "warm": []}

    # Phase 1: Cold start (第一次请求每个 agent)
    print(f"\n📊 Phase 1: Cold Start (cache miss)")
    print("-" * 80)

    for agent_name, system_prompt in agent_system_prompts.items():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_queries[0]}
        ]

        ttft = measure_ttft(api_url, messages)
        results["cold"].append(ttft)
        print(f"  {agent_name}: TTFT = {ttft:.1f}ms (cache miss)")

    cold_avg = sum(results["cold"]) / len(results["cold"])
    print(f"\n⭐ Cold Start Avg TTFT: {cold_avg:.1f}ms")

    # Phase 2: Warm cache (每个 agent 再次请求，不同 user query)
    print(f"\n📊 Phase 2: Warm Cache (cache hit)")
    print("-" * 80)

    for agent_name, system_prompt in agent_system_prompts.items():
        for query in user_queries[1:queries_per_agent+1]:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]

            ttft = measure_ttft(api_url, messages)
            results["warm"].append(ttft)
            print(f"  {agent_name} + '{query[:30]}...': TTFT = {ttft:.1f}ms (cache hit)")

    warm_avg = sum(results["warm"]) / len(results["warm"])
    print(f"\n⭐ Warm Cache Avg TTFT: {warm_avg:.1f}ms")

    # Results
    print("\n" + "=" * 80)
    print("📈 Results Summary")
    print("=" * 80)

    print(f"\n⭐ Cold Start (cache miss):")
    print(f"   Avg TTFT: {cold_avg:.1f}ms")
    print(f"   Samples: {len(results['cold'])}")

    print(f"\n⭐ Warm Cache (cache hit):")
    print(f"   Avg TTFT: {warm_avg:.1f}ms")
    print(f"   Samples: {len(results['warm'])}")

    improvement = (cold_avg - warm_avg) / cold_avg * 100
    time_saved = (cold_avg - warm_avg)

    print(f"\n📊 Improvement:")
    print(f"   TTFT Reduction: {improvement:.1f}%")
    print(f"   Time Saved: {time_saved:.1f}ms per request")

    if improvement > 50:
        print(f"\n✅ Excellent! Prefix Caching provides {improvement:.1f}% TTFT improvement! ⭐⭐⭐")
        print("   ThunderOMLX's BlockAwarePrefixCache is working correctly!")
    elif improvement > 30:
        print(f"\n✅ Good! Prefix Caching provides {improvement:.1f}% improvement! ⭐⭐")
    elif improvement > 10:
        print(f"\n✅ Moderate improvement ({improvement:.1f}%)")
    elif improvement > 0:
        print(f"\n⚠️  Limited improvement ({improvement:.1f}%)")
        print("   Check if paged_ssd_cache_dir is set correctly")
    else:
        print(f"\n❌ No improvement or regression ({improvement:.1f}%)")
        print("   Possible issues:")
        print("   1. Cache is disabled (--paged-ssd-cache-dir not set)")
        print("   2. System prompts are not identical (ContextPilot mismatch)")
        print("   3. Cache is cold (first run after server restart)")

    # Fetch final cache stats
    print("\n" + "=" * 80)
    print("📊 Cache Statistics")
    print("=" * 80)

    try:
        stats_response = requests.get(f"{api_url.replace('/v1/chat/completions', '/cache/stats')}")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"\n{json.dumps(stats, indent=2)}")
        else:
            print(f"\n⚠️  Could not fetch cache stats (status code: {stats_response.status_code})")
    except Exception as e:
        print(f"\n⚠️  Could not fetch cache stats: {e}")

    print("\n" + "=" * 80)

    return cold_avg, warm_avg, improvement


def main():
    parser = argparse.ArgumentParser(
        description="ThunderOMLX Prefix Caching Benchmark (HTTP API)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8080/v1/chat/completions",
        help="ThunderOMLX API endpoint",
    )
    parser.add_argument(
        "--system-prompt-length",
        type=int,
        default=800,
        help="System prompt length in tokens",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="Number of agent types (OpenClaw scenario)",
    )
    parser.add_argument(
        "--queries-per-agent",
        type=int,
        default=3,
        help="Number of queries per agent (warm cache phase)",
    )

    args = parser.parse_args()

    benchmark_openclaw_workload(
        api_url=args.api_url,
        system_prompt_length=args.system_prompt_length,
        num_agents=args.num_agents,
        queries_per_agent=args.queries_per_agent,
    )


if __name__ == "__main__":
    main()
