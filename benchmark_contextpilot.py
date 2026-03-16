#!/usr/bin/env python3
"""
ContextPilot Phase 2 Benchmark
基于 OpenClaw 真实 workload 数据，测试 ContextPilot 集成后的 cache 命中效果。

测试方法：
1. 从 workload 采样 20 个请求（保持 agent 分布）
2. 每个请求使用对应 agent 的 system prompt + 不同 user query
3. 串行发送（避免 OOM），测量 TTFT、TPS、cache hit
4. 重点观察：相同 agent 连续请求的 cache 复用
"""
import asyncio
import json
import time
import random
import sys
from pathlib import Path

# Agent system prompts (模拟 OpenClaw 的固定 system prompt)
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

# 每个 agent 的 user query 样本
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


def sample_workload(n=20, seed=42):
    """从 workload 数据中采样，模拟真实 agent 调用序列"""
    random.seed(seed)

    # 加载 workload 获取 agent 分布
    workload_path = Path(__file__).parent / "openclaw-workload" / "openclaw-workload-7d.jsonl"
    agents_in_workload = []
    with open(workload_path) as f:
        for line in f:
            d = json.loads(line)
            agents_in_workload.append(d["agent_id"])

    # 按分布采样
    sampled_agents = random.choices(agents_in_workload, k=n)

    # 为每个 agent 分配 query
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


async def run_benchmark(requests, port=8000):
    """串行执行请求，收集性能数据"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="not-needed",
    )

    # 确认模型可用
    models = await client.models.list()
    if not models.data:
        print("No models available!")
        return []
    model_id = models.data[0].id
    print(f"Model: {model_id}")

    results = []
    for i, req in enumerate(requests):
        agent_id = req["agent_id"]
        sys_prompt = req["system_prompt"]
        user_query = req["user_query"]

        print(f"\n[{i+1}/{len(requests)}] {agent_id}: {user_query[:60]}...")

        start = time.perf_counter()
        first_token_time = None
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = None

        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_query},
                ],
                max_tokens=50,
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                if hasattr(chunk, "usage") and chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    cached_tokens = getattr(chunk.usage, "cached_tokens", None)

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "agent_id": agent_id,
                "error": str(e),
            })
            continue

        end = time.perf_counter()
        if first_token_time is None:
            first_token_time = end

        ttft_ms = (first_token_time - start) * 1000
        total_ms = (end - start) * 1000
        gen_tps = completion_tokens / (end - first_token_time) if (end - first_token_time) > 0 else 0

        result = {
            "idx": i,
            "agent_id": agent_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "ttft_ms": round(ttft_ms, 1),
            "total_ms": round(total_ms, 1),
            "gen_tps": round(gen_tps, 1),
        }
        results.append(result)

        cache_info = f", cached={cached_tokens}" if cached_tokens else ""
        print(f"  TTFT={ttft_ms:.0f}ms, TPS={gen_tps:.1f}, prompt={prompt_tokens}, gen={completion_tokens}{cache_info}")

    return results


def print_summary(results):
    """打印汇总报告"""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\nNo valid results!")
        return

    print("\n" + "=" * 70)
    print("  ContextPilot Phase 2 Benchmark Report")
    print("=" * 70)

    # 总体指标
    avg_ttft = sum(r["ttft_ms"] for r in valid) / len(valid)
    avg_tps = sum(r["gen_tps"] for r in valid) / len(valid)
    total_prompt = sum(r["prompt_tokens"] for r in valid)
    total_gen = sum(r["completion_tokens"] for r in valid)
    total_cached = sum(r["cached_tokens"] or 0 for r in valid)

    print(f"\n  Total requests: {len(valid)}")
    print(f"  Avg TTFT: {avg_ttft:.0f} ms")
    print(f"  Avg Gen TPS: {avg_tps:.1f} tok/s")
    print(f"  Total prompt tokens: {total_prompt}")
    print(f"  Total gen tokens: {total_gen}")
    print(f"  Total cached tokens: {total_cached}")
    if total_prompt > 0:
        print(f"  Cache hit ratio: {total_cached / total_prompt * 100:.1f}%")

    # 按 Agent 分组
    from collections import defaultdict
    by_agent = defaultdict(list)
    for r in valid:
        by_agent[r["agent_id"]].append(r)

    print(f"\n  Per-Agent Breakdown:")
    print(f"  {'Agent':<20} {'Count':>5} {'Avg TTFT':>10} {'Avg TPS':>10} {'Cache%':>8}")
    print(f"  {'-'*20} {'-'*5} {'-'*10} {'-'*10} {'-'*8}")

    for agent_id in sorted(by_agent.keys()):
        items = by_agent[agent_id]
        a_ttft = sum(r["ttft_ms"] for r in items) / len(items)
        a_tps = sum(r["gen_tps"] for r in items) / len(items)
        a_prompt = sum(r["prompt_tokens"] for r in items)
        a_cached = sum(r["cached_tokens"] or 0 for r in items)
        a_cache_pct = (a_cached / a_prompt * 100) if a_prompt > 0 else 0
        print(f"  {agent_id:<20} {len(items):>5} {a_ttft:>8.0f}ms {a_tps:>8.1f} {a_cache_pct:>7.1f}%")

    # 逐请求对比（看 cache 复用效果）
    print(f"\n  Request Timeline (cache reuse tracking):")
    print(f"  {'#':>3} {'Agent':<20} {'Prompt':>6} {'Cached':>7} {'TTFT':>8} {'TPS':>8}")
    print(f"  {'-'*3} {'-'*20} {'-'*6} {'-'*7} {'-'*8} {'-'*8}")

    for r in valid:
        cached = r["cached_tokens"] or 0
        cached_pct = f"({cached/r['prompt_tokens']*100:.0f}%)" if r["prompt_tokens"] > 0 and cached > 0 else ""
        print(f"  {r['idx']:>3} {r['agent_id']:<20} {r['prompt_tokens']:>6} {cached:>5}{cached_pct:>5} {r['ttft_ms']:>6.0f}ms {r['gen_tps']:>6.1f}")

    print("\n" + "=" * 70)


async def main():
    print("OpenClaw ContextPilot Benchmark")
    print("=" * 40)

    # 采样 workload
    requests = sample_workload(n=20)

    # 打印采样分布
    from collections import Counter
    dist = Counter(r["agent_id"] for r in requests)
    print(f"\nSampled {len(requests)} requests:")
    for agent, count in dist.most_common():
        print(f"  {agent}: {count}")

    # 运行 benchmark
    results = await run_benchmark(requests)

    # 保存原始数据
    output_path = Path(__file__).parent / "benchmark_contextpilot_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # 打印汇总
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
