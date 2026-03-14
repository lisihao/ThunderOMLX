#!/usr/bin/env python3
"""
Benchmark oMLX performance to compare with ThunderLLAMA baseline.

ThunderLLAMA baseline (Agent scenario):
- Model: Qwen3-30B-A3B (Q5_K_M quantization)
- Context: 4 concurrent requests, ~1024 tokens each
- Throughput: 687.6 tok/s
- Cache hit: 99.7%
- Skip rate: 94%
"""
import asyncio
import json
import time
from openai import AsyncOpenAI


async def run_concurrent_requests(client, model_id: str, num_concurrent: int = 4):
    """Run concurrent requests to simulate Agent scenario."""

    # Fixed system prompt (similar to Agent scenario)
    system_prompt = """You are a helpful AI assistant. You provide clear, accurate, and concise answers."""

    # Generate prompts with slight variations
    prompts = [
        "Explain the key differences between Python and JavaScript in detail.",
        "What are the main advantages of using TypeScript over JavaScript?",
        "Describe the most important design patterns in software engineering.",
        "Explain how asynchronous programming works in Python with examples."
    ]

    async def single_request(prompt: str):
        """Run a single completion request."""
        start = time.perf_counter()
        completion_tokens = 0
        prompt_tokens = 0
        first_token_time = None

        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=128,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True}
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

            # Get token counts from usage
            if hasattr(chunk, 'usage') and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

        end = time.perf_counter()

        if first_token_time is None:
            first_token_time = end

        return {
            "ttft_s": first_token_time - start,
            "total_time_s": end - start,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

    # Run all requests concurrently
    print(f"\n🚀 Running {num_concurrent} concurrent requests...")
    wall_start = time.perf_counter()

    results = await asyncio.gather(*[
        single_request(prompts[i % len(prompts)])
        for i in range(num_concurrent)
    ])

    wall_end = time.perf_counter()
    wall_time = wall_end - wall_start

    # Aggregate metrics
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    avg_ttft_s = sum(r["ttft_s"] for r in results) / len(results)

    # Generation throughput (tok/s)
    gen_tps = total_completion_tokens / wall_time

    # Prefill throughput
    max_ttft = max(r["ttft_s"] for r in results)
    pp_tps = total_prompt_tokens / max_ttft if max_ttft > 0 else 0

    return {
        "batch_size": num_concurrent,
        "wall_time_s": wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_ttft_ms": avg_ttft_s * 1000,
        "pp_tps": pp_tps,
        "gen_tps": gen_tps,
        "total_throughput": (total_prompt_tokens + total_completion_tokens) / wall_time
    }


async def run_single_request_benchmark(client, model_id: str):
    """Run single request benchmark at different context lengths."""
    print("\n📊 Running single request benchmarks...")

    results = []

    # Test different context lengths (matching oMLX benchmark)
    test_configs = [
        {"context": 1024, "max_tokens": 128},
        {"context": 4096, "max_tokens": 128},
    ]

    for config in test_configs:
        context_len = config["context"]
        max_tokens = config["max_tokens"]

        print(f"\n  Testing: pp{context_len}/tg{max_tokens}")

        # Generate prompt with target token count
        prompt = "The quick brown fox jumps over the lazy dog. " * (context_len // 10)

        start = time.perf_counter()
        first_token_time = None
        completion_tokens = 0
        prompt_tokens = 0

        response = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True}
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

            if hasattr(chunk, 'usage') and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

        end = time.perf_counter()

        if first_token_time is None:
            first_token_time = end

        ttft_s = first_token_time - start
        gen_time_s = end - first_token_time

        result = {
            "pp": prompt_tokens,
            "tg": completion_tokens,
            "ttft_ms": ttft_s * 1000,
            "gen_tps": completion_tokens / gen_time_s if gen_time_s > 0 else 0,
            "pp_tps": prompt_tokens / ttft_s if ttft_s > 0 else 0,
            "e2e_s": end - start
        }

        results.append(result)

        print(f"    TTFT: {result['ttft_ms']:.1f}ms")
        print(f"    Gen TPS: {result['gen_tps']:.1f} tok/s")
        print(f"    PP TPS: {result['pp_tps']:.1f} tok/s")

    return results


async def main():
    """Main benchmark entry point."""

    # Connect to oMLX server
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="not-needed"
    )

    # Get available models
    models = await client.models.list()
    if not models.data:
        print("❌ No models available")
        return

    model_id = models.data[0].id
    print(f"✅ Using model: {model_id}")

    # Run single request benchmarks
    single_results = await run_single_request_benchmark(client, model_id)

    # Run concurrent benchmark (Agent scenario)
    concurrent_result = await run_concurrent_requests(client, model_id, num_concurrent=4)

    print("\n" + "="*60)
    print("📈 BENCHMARK RESULTS")
    print("="*60)

    print("\n🔷 Single Request Performance:")
    for r in single_results:
        print(f"  pp{r['pp']}/tg{r['tg']}: {r['gen_tps']:.1f} tok/s (TTFT: {r['ttft_ms']:.1f}ms)")

    print("\n🔷 Agent Scenario (4 Concurrent):")
    print(f"  Wall time: {concurrent_result['wall_time_s']:.2f}s")
    print(f"  Total tokens: {concurrent_result['total_completion_tokens']} generated")
    print(f"  Generation TPS: {concurrent_result['gen_tps']:.1f} tok/s")
    print(f"  Prefill TPS: {concurrent_result['pp_tps']:.1f} tok/s")
    print(f"  Avg TTFT: {concurrent_result['avg_ttft_ms']:.1f}ms")

    print("\n🔷 ThunderLLAMA Baseline (for comparison):")
    print(f"  Model: Qwen3-30B-A3B-Q5_K_M")
    print(f"  Generation TPS: 687.6 tok/s")
    print(f"  Cache hit: 99.7%")
    print(f"  Skip rate: 94%")

    print("\n🔷 Performance Ratio:")
    ratio = concurrent_result['gen_tps'] / 687.6
    print(f"  oMLX / ThunderLLAMA: {ratio:.2f}x ({ratio*100:.1f}%)")

    if ratio < 0.5:
        print(f"\n⚠️  oMLX is {1/ratio:.1f}x slower - significant optimization potential!")
        print(f"  Estimated gain from ThunderLLAMA features: {687.6 - concurrent_result['gen_tps']:.1f} tok/s")
    elif ratio < 0.8:
        print(f"\n📊 oMLX is moderately slower - good opportunity for optimization")
    else:
        print(f"\n✅ oMLX performance is competitive")

    # Save detailed results
    output = {
        "model": model_id,
        "timestamp": time.time(),
        "single_request": single_results,
        "agent_scenario": concurrent_result,
        "thunderllama_baseline": {
            "gen_tps": 687.6,
            "cache_hit": 0.997,
            "skip_rate": 0.94
        },
        "performance_ratio": ratio
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Detailed results saved to: benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
