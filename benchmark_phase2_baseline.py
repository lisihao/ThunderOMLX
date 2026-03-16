#!/usr/bin/env python3
"""
ContextPilot Phase 2 性能基线测试

目标：在启动 Phase 3 之前建立性能基线数据。

测试场景：
  Test 1: 多轮对话 — 相同 system prompt，不同 user query（验证 prefix cache 复用）
  Test 2: OpenClaw workload — 真实 agent 场景（20 请求采样）

指标：
  - TTFT (Time To First Token) — 通过 streaming 测量
  - cache_hit — 通过日志 + 内部 metrics 验证
  - message_boundaries — ContextPilot adapter 的 token 边界
  - gen_tps — 生成速度

用法：
  /Users/lisihao/ThunderOMLX/venv/bin/python benchmark_phase2_baseline.py [--test 1|2|all] [--rounds N]
"""

import asyncio
import json
import time
import sys
import os
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from openai import AsyncOpenAI

# ContextPilot adapter
from omlx.contextpilot.adapter import ContextPilotAdapter, ContextIndex

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

BASE_URL = "http://127.0.0.1:8000/v1"
MODEL = "qwen3.5-35b-mlx"
MAX_TOKENS = 30
TEMPERATURE = 0.0

# Shared system prompt (simulating multi-turn conversation with same agent)
SYSTEM_PROMPT = (
    "You are a senior software engineer specializing in Python, TypeScript, and Go. "
    "You write clean, maintainable, and well-tested code. Follow SOLID principles. "
    "Always include error handling and type annotations. "
    "When reviewing code, check for security vulnerabilities, performance issues, and code smells. "
    "Provide specific, actionable suggestions with code examples. "
    "Use industry best practices and design patterns appropriate for the context. "
    "Consider edge cases and potential failure modes in your implementations. "
    "Always explain your reasoning and trade-offs when making architectural decisions. "
    "Prefer composition over inheritance. Keep functions small and focused. "
    "Write comprehensive tests including unit tests, integration tests, and edge cases."
)

MULTI_TURN_QUERIES = [
    "What is the time complexity of a hash table lookup?",
    "How do I implement a thread-safe singleton in Python?",
    "Explain the difference between asyncio.gather and asyncio.wait.",
    "What are the best practices for error handling in Go?",
    "How should I design a retry mechanism with exponential backoff?",
    "What is the observer pattern and when should I use it?",
    "How do I optimize a slow SQL query that scans millions of rows?",
    "What is the difference between ACID and BASE in databases?",
    "How should I structure a microservices project?",
    "What are the security implications of using JWT tokens?",
]


@dataclass
class RequestResult:
    idx: int
    test_name: str
    agent_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    gen_tps: float = 0.0
    error: Optional[str] = None
    # ContextPilot metadata
    message_boundaries: List[int] = field(default_factory=list)
    context_refs_count: int = 0
    common_prefix_len: int = 0


# ============================================================================
# Test 1: Multi-Turn Dialog (same system prompt, different queries)
# ============================================================================

async def test_multi_turn_dialog(
    client: AsyncOpenAI,
    adapter: ContextPilotAdapter,
    rounds: int = 2,
) -> List[RequestResult]:
    """
    Test 1: 多轮对话
    - 相同 system prompt（模拟同一 agent 的连续调用）
    - 不同 user query
    - 验证 prefix cache 在 system prompt 部分的复用
    """
    print("\n" + "=" * 70)
    print("  TEST 1: Multi-Turn Dialog (same system prompt)")
    print(f"  Rounds: {rounds}, Queries per round: {len(MULTI_TURN_QUERIES)}")
    print("=" * 70)

    results: List[RequestResult] = []
    previous_requests: List[List[Dict[str, str]]] = []

    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1}/{rounds} ---")

        for qi, query in enumerate(MULTI_TURN_QUERIES):
            idx = round_num * len(MULTI_TURN_QUERIES) + qi
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]

            # Run ContextPilot optimization
            optimized = adapter.optimize_request(messages, previous_requests=previous_requests)
            msg_boundaries = optimized["message_boundaries"]
            ctx_refs_count = len(optimized["context_refs"])

            # Track previous requests for prefix detection
            previous_requests.append(messages)
            if len(previous_requests) > 10:
                previous_requests = previous_requests[-10:]

            # Send request with streaming to measure TTFT
            result = await send_streaming_request(
                client, messages, idx, "multi-turn", "coder-agent"
            )
            result.message_boundaries = msg_boundaries
            result.context_refs_count = ctx_refs_count
            results.append(result)

            # Print result
            cache_str = f"cached={result.cached_tokens}" if result.cached_tokens else "no-cache"
            print(
                f"  [{qi+1:2d}] TTFT={result.ttft_ms:6.0f}ms "
                f"TPS={result.gen_tps:5.1f} "
                f"prompt={result.prompt_tokens:4d} "
                f"{cache_str} "
                f"refs={ctx_refs_count}"
            )

    return results


# ============================================================================
# Test 2: OpenClaw Workload (real agent scenario)
# ============================================================================

AGENT_SYSTEM_PROMPTS = {
    "coder-agent": (
        "You are a senior software engineer specializing in Python, TypeScript, and Go. "
        "You write clean, maintainable, and well-tested code. Follow SOLID principles. "
        "Always include error handling and type annotations. "
        "When reviewing code, check for security vulnerabilities, performance issues, and code smells. "
        "Provide specific, actionable suggestions with code examples. "
        "Use industry best practices and design patterns appropriate for the context. "
        "Consider edge cases and potential failure modes in your implementations. "
        "Always explain your reasoning and trade-offs when making architectural decisions. "
        "Prefer composition over inheritance. Keep functions small and focused. "
        "Write comprehensive tests including unit tests, integration tests, and edge cases."
    ),
    "researcher-agent": (
        "You are a research analyst with expertise in technology, market trends, and competitive analysis. "
        "You provide thorough, well-sourced analysis with clear conclusions and actionable recommendations. "
        "Structure your research with executive summary, methodology, findings, and recommendations. "
        "Use quantitative data when available and cite sources. "
        "Identify potential biases in data and analysis. "
        "Consider multiple perspectives and present balanced viewpoints. "
        "Highlight areas of uncertainty and suggest further research when needed. "
        "Provide practical, implementable recommendations prioritized by impact and feasibility. "
        "Compare alternatives using consistent criteria and scoring frameworks. "
        "Include relevant market data, industry benchmarks, and case studies to support your analysis. "
        "Summarize key findings and next steps at the end of each analysis."
    ),
    "analyst-agent": (
        "You are a data analyst specializing in performance metrics, system optimization, and data-driven decision making. "
        "You analyze data with statistical rigor and present findings clearly. "
        "Use appropriate statistical methods and visualizations. "
        "Identify trends, anomalies, and correlations in data. "
        "Provide confidence intervals and significance tests where applicable. "
        "Translate technical findings into business-relevant insights. "
        "Always validate data quality before analysis. "
        "Consider confounding variables and selection bias."
    ),
    "pm-agent": (
        "You are a product manager with experience in agile development and user-centered design. "
        "You balance technical feasibility with business value and user needs. "
        "Write clear user stories with acceptance criteria. "
        "Prioritize features using frameworks like RICE or MoSCoW. "
        "Define success metrics and KPIs for each feature."
    ),
    "tester-agent": (
        "You are a QA engineer specializing in test strategy, automation, and quality assurance. "
        "You design comprehensive test plans covering functional, integration, performance, and security testing. "
        "Use risk-based testing to prioritize test coverage. "
        "Write clear, reproducible test cases with expected results. "
        "Identify edge cases and boundary conditions. "
        "Recommend appropriate testing tools and frameworks."
    ),
}

AGENT_QUERIES = {
    "coder-agent": [
        "Implement a thread-safe LRU cache in Python with O(1) operations.",
        "Review this async handler and suggest improvements for error handling.",
        "Write a migration script to add indexes to the users table.",
        "Optimize this database query that scans 1M rows.",
        "Create a retry decorator with exponential backoff.",
    ],
    "researcher-agent": [
        "Compare MLX vs PyTorch for Apple Silicon inference performance.",
        "Analyze the competitive landscape for local LLM inference engines.",
        "Research the latest advances in KV cache optimization for transformers.",
        "Survey open-source projects implementing prefix caching.",
    ],
    "analyst-agent": [
        "Analyze the cache hit rate trends from the last 7 days of data.",
        "What is the performance impact of block size on prefill latency?",
        "Compare the throughput of different quantization levels.",
        "Identify bottlenecks in the current request processing pipeline.",
        "Calculate the expected ROI of implementing message-level caching.",
    ],
    "pm-agent": [
        "Write a PRD for the ContextPilot feature integration.",
        "Define success metrics for the cache optimization project.",
        "Prioritize the Phase 3 feature backlog.",
    ],
    "tester-agent": [
        "Design a test plan for the prefix cache system.",
        "Write integration tests for the ContextPilot adapter.",
        "Create a benchmark suite for measuring TTFT improvements.",
    ],
}


def sample_openclaw_workload(n=20, seed=42):
    """从 OpenClaw workload 采样请求序列"""
    import random
    random.seed(seed)

    workload_path = Path(__file__).parent / "openclaw-workload" / "openclaw-workload-7d.jsonl"
    if not workload_path.exists():
        # Fallback: generate synthetic workload
        print("  (workload file not found, using synthetic distribution)")
        agents = list(AGENT_SYSTEM_PROMPTS.keys())
        weights = [0.35, 0.25, 0.20, 0.10, 0.10]  # coder heavy
        sampled_agents = random.choices(agents, weights=weights, k=n)
    else:
        agents_in_workload = []
        with open(workload_path) as f:
            for line in f:
                d = json.loads(line)
                agents_in_workload.append(d["agent_id"])
        sampled_agents = random.choices(agents_in_workload, k=n)

    query_idx = {agent: 0 for agent in AGENT_SYSTEM_PROMPTS}
    requests = []
    for agent_id in sampled_agents:
        queries = AGENT_QUERIES[agent_id]
        idx = query_idx[agent_id] % len(queries)
        query_idx[agent_id] += 1
        requests.append({
            "agent_id": agent_id,
            "system_prompt": AGENT_SYSTEM_PROMPTS[agent_id],
            "user_query": queries[idx],
        })

    return requests


async def test_openclaw_workload(
    client: AsyncOpenAI,
    adapter: ContextPilotAdapter,
    n: int = 20,
) -> List[RequestResult]:
    """
    Test 2: OpenClaw workload
    - 多种 agent，各有固定 system prompt
    - 相同 agent 连续出现时应该有缓存复用
    """
    print("\n" + "=" * 70)
    print("  TEST 2: OpenClaw Workload (real agent scenario)")
    print(f"  Requests: {n}")
    print("=" * 70)

    workload = sample_openclaw_workload(n=n)

    # Print distribution
    from collections import Counter
    dist = Counter(r["agent_id"] for r in workload)
    print(f"\n  Agent distribution:")
    for agent, count in dist.most_common():
        print(f"    {agent}: {count}")

    results: List[RequestResult] = []
    previous_requests: List[List[Dict[str, str]]] = []

    for i, req in enumerate(workload):
        messages = [
            {"role": "system", "content": req["system_prompt"]},
            {"role": "user", "content": req["user_query"]},
        ]

        # ContextPilot optimization
        optimized = adapter.optimize_request(messages, previous_requests=previous_requests)
        msg_boundaries = optimized["message_boundaries"]
        ctx_refs_count = len(optimized["context_refs"])

        previous_requests.append(messages)
        if len(previous_requests) > 20:
            previous_requests = previous_requests[-20:]

        result = await send_streaming_request(
            client, messages, i, "openclaw", req["agent_id"]
        )
        result.message_boundaries = msg_boundaries
        result.context_refs_count = ctx_refs_count
        results.append(result)

        cache_str = f"cached={result.cached_tokens}" if result.cached_tokens else "no-cache"
        short_agent = req["agent_id"].replace("-agent", "")
        print(
            f"  [{i+1:2d}] {short_agent:>10s} "
            f"TTFT={result.ttft_ms:6.0f}ms "
            f"TPS={result.gen_tps:5.1f} "
            f"prompt={result.prompt_tokens:4d} "
            f"{cache_str}"
        )

    return results


# ============================================================================
# Streaming Request Helper
# ============================================================================

async def send_streaming_request(
    client: AsyncOpenAI,
    messages: List[Dict[str, str]],
    idx: int,
    test_name: str,
    agent_id: str,
) -> RequestResult:
    """Send a streaming request and measure TTFT precisely."""
    result = RequestResult(idx=idx, test_name=test_name, agent_id=agent_id)

    start = time.perf_counter()
    first_token_time = None

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
            if hasattr(chunk, "usage") and chunk.usage:
                result.prompt_tokens = chunk.usage.prompt_tokens or 0
                result.completion_tokens = chunk.usage.completion_tokens or 0
                cached = getattr(chunk.usage, "cached_tokens", None)
                result.cached_tokens = cached or 0

    except Exception as e:
        result.error = str(e)
        return result

    end = time.perf_counter()
    if first_token_time is None:
        first_token_time = end

    result.ttft_ms = round((first_token_time - start) * 1000, 1)
    result.total_ms = round((end - start) * 1000, 1)

    gen_duration = end - first_token_time
    if gen_duration > 0 and result.completion_tokens > 0:
        result.gen_tps = round(result.completion_tokens / gen_duration, 1)

    return result


# ============================================================================
# Report Generation
# ============================================================================

def print_report(all_results: Dict[str, List[RequestResult]]):
    """Print comprehensive baseline report."""
    print("\n" + "=" * 70)
    print("  CONTEXTPILOT PHASE 2 - PERFORMANCE BASELINE REPORT")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: {MODEL}")
    print("=" * 70)

    for test_name, results in all_results.items():
        valid = [r for r in results if r.error is None]
        if not valid:
            print(f"\n  [{test_name}] No valid results!")
            continue

        print(f"\n  --- {test_name} ---")
        print(f"  Total requests: {len(valid)}")

        # TTFT stats
        ttfts = [r.ttft_ms for r in valid]
        avg_ttft = sum(ttfts) / len(ttfts)
        min_ttft = min(ttfts)
        max_ttft = max(ttfts)

        # Separate cold (first request) vs warm
        cold_ttft = ttfts[0] if ttfts else 0
        warm_ttfts = ttfts[1:] if len(ttfts) > 1 else []
        avg_warm_ttft = sum(warm_ttfts) / len(warm_ttfts) if warm_ttfts else 0

        print(f"\n  TTFT (Time To First Token):")
        print(f"    Cold (1st request):  {cold_ttft:.0f} ms")
        print(f"    Avg (warm):          {avg_warm_ttft:.0f} ms")
        print(f"    Avg (all):           {avg_ttft:.0f} ms")
        print(f"    Min / Max:           {min_ttft:.0f} / {max_ttft:.0f} ms")
        if cold_ttft > 0 and avg_warm_ttft > 0:
            print(f"    Speedup (cold→warm): {cold_ttft / avg_warm_ttft:.1f}x")

        # TPS stats
        tps_vals = [r.gen_tps for r in valid if r.gen_tps > 0]
        if tps_vals:
            avg_tps = sum(tps_vals) / len(tps_vals)
            print(f"\n  Generation TPS:")
            print(f"    Average: {avg_tps:.1f} tok/s")

        # Cache stats
        total_prompt = sum(r.prompt_tokens for r in valid)
        total_cached = sum(r.cached_tokens for r in valid)
        cache_pct = (total_cached / total_prompt * 100) if total_prompt > 0 else 0
        print(f"\n  Cache:")
        print(f"    Total prompt tokens:  {total_prompt}")
        print(f"    Total cached tokens:  {total_cached}")
        print(f"    Cache hit ratio:      {cache_pct:.1f}%")

        # ContextPilot metadata
        refs_total = sum(r.context_refs_count for r in valid)
        print(f"\n  ContextPilot:")
        print(f"    Total context refs:   {refs_total}")
        print(f"    Avg refs/request:     {refs_total / len(valid):.1f}")

        # Per-agent breakdown
        by_agent = defaultdict(list)
        for r in valid:
            by_agent[r.agent_id].append(r)

        if len(by_agent) > 1:
            print(f"\n  Per-Agent Breakdown:")
            print(f"  {'Agent':<18} {'N':>3} {'Avg TTFT':>10} {'Avg TPS':>10} {'Cache%':>8}")
            print(f"  {'-'*18} {'-'*3} {'-'*10} {'-'*10} {'-'*8}")

            for agent_id in sorted(by_agent.keys()):
                items = by_agent[agent_id]
                a_ttft = sum(r.ttft_ms for r in items) / len(items)
                a_tps_vals = [r.gen_tps for r in items if r.gen_tps > 0]
                a_tps = sum(a_tps_vals) / len(a_tps_vals) if a_tps_vals else 0
                a_prompt = sum(r.prompt_tokens for r in items)
                a_cached = sum(r.cached_tokens for r in items)
                a_cache_pct = (a_cached / a_prompt * 100) if a_prompt > 0 else 0
                short = agent_id.replace("-agent", "")
                print(f"  {short:<18} {len(items):>3} {a_ttft:>8.0f}ms {a_tps:>8.1f} {a_cache_pct:>7.1f}%")

        # Request timeline
        print(f"\n  Request Timeline:")
        print(f"  {'#':>3} {'Agent':<12} {'Prompt':>6} {'Cached':>7} {'TTFT':>8} {'TPS':>8}")
        print(f"  {'-'*3} {'-'*12} {'-'*6} {'-'*7} {'-'*8} {'-'*8}")

        for r in valid:
            cached_str = str(r.cached_tokens) if r.cached_tokens > 0 else "-"
            short = r.agent_id.replace("-agent", "")
            print(
                f"  {r.idx:>3} {short:<12} "
                f"{r.prompt_tokens:>6} {cached_str:>7} "
                f"{r.ttft_ms:>6.0f}ms {r.gen_tps:>6.1f}"
            )

    print("\n" + "=" * 70)


def save_results(all_results: Dict[str, List[RequestResult]], output_path: Path):
    """Save raw results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "tests": {},
    }

    for test_name, results in all_results.items():
        data["tests"][test_name] = [asdict(r) for r in results]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n  Raw results saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="ContextPilot Phase 2 Baseline Benchmark")
    parser.add_argument("--test", choices=["1", "2", "all"], default="all", help="Which test to run")
    parser.add_argument("--rounds", type=int, default=2, help="Rounds for multi-turn test")
    parser.add_argument("--n", type=int, default=20, help="Requests for OpenClaw test")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = f"http://127.0.0.1:{args.port}/v1"

    # Check server health
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as hc:
            resp = await hc.get(f"http://127.0.0.1:{args.port}/health")
            health = resp.json()
            print(f"Server: {health.get('status', 'unknown')}")
            print(f"Default model: {health.get('default_model', 'unknown')}")
    except Exception as e:
        print(f"Server not available on port {args.port}: {e}")
        return

    client = AsyncOpenAI(base_url=BASE_URL, api_key="not-needed")
    context_index = ContextIndex()
    adapter = ContextPilotAdapter(context_index)

    all_results: Dict[str, List[RequestResult]] = {}

    if args.test in ("1", "all"):
        results = await test_multi_turn_dialog(client, adapter, rounds=args.rounds)
        all_results["multi-turn"] = results

    if args.test in ("2", "all"):
        results = await test_openclaw_workload(client, adapter, n=args.n)
        all_results["openclaw"] = results

    # Print report
    print_report(all_results)

    # Save results
    output_path = Path(__file__).parent / "benchmark_phase2_baseline_results.json"
    save_results(all_results, output_path)

    # Print ContextPilot index stats
    print(f"\n  ContextPilot Index: {len(context_index)} unique blocks indexed")


if __name__ == "__main__":
    asyncio.run(main())
