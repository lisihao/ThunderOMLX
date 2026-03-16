#!/usr/bin/env python3
"""
Profile ThunderOMLX server API performance.
"""

import cProfile
import pstats
import io
import time
import asyncio
from openai import AsyncOpenAI

async def profile_server_api():
    """Profile ThunderOMLX server API calls."""

    print("=" * 80)
    print("🔍 ThunderOMLX Server API Performance Profiling")
    print("=" * 80)

    # Connect to ThunderOMLX server
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="not-needed"
    )

    # Prepare test prompt (2048 tokens equivalent)
    prompt = "The quick brown fox jumps over the lazy dog. " * 200

    print(f"\n📝 Prompt: ~{len(prompt.split())} words (~2048 tokens)")
    print("🎯 Target: 128 generation tokens")

    # Profile API call
    print("\n🚀 Starting profiled API call...")

    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()
    ttft = None
    tokens_generated = 0

    # Stream API call
    response = await client.chat.completions.create(
        model="qwen3.5-35b-mlx",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True}
    )

    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - start

        if hasattr(chunk, 'usage') and chunk.usage:
            tokens_generated = chunk.usage.completion_tokens

    end = time.perf_counter()
    profiler.disable()

    # Calculate metrics
    total_time = end - start
    gen_time = total_time - ttft if ttft else total_time
    gen_tps = tokens_generated / gen_time if gen_time > 0 else 0

    print(f"\n✅ API call completed")
    print(f"  TTFT: {ttft*1000:.1f}ms" if ttft else "  TTFT: N/A")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Tokens generated: {tokens_generated}")
    print(f"  Generation TPS: {gen_tps:.1f} tok/s")

    # Analyze profile
    print("\n" + "=" * 80)
    print("📊 CLIENT-SIDE PROFILE - Top 30 Functions by Cumulative Time")
    print("=" * 80)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)

    profile_output = s.getvalue()
    print(profile_output)

    # Save to file
    with open("server_api_profile.txt", "w") as f:
        f.write(f"ThunderOMLX Server API Profile\n")
        f.write(f"TTFT: {ttft*1000:.1f}ms\n" if ttft else "TTFT: N/A\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Generation TPS: {gen_tps:.1f} tok/s\n")
        f.write(f"\n{profile_output}")

    print("\n💾 Profile saved to: server_api_profile.txt")

    # Analyze by tottime
    print("\n" + "=" * 80)
    print("📊 Top 20 Functions by Self Time")
    print("=" * 80)

    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
    ps2.print_stats(20)

    print(s2.getvalue())

    print("\n" + "=" * 80)
    print("✅ API Profiling complete")
    print("=" * 80)

    return gen_tps

async def run_multiple_requests():
    """Run multiple requests to get average performance."""
    print("\n" + "=" * 80)
    print("🔄 Running 3 requests for average performance")
    print("=" * 80)

    client = AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="not-needed"
    )

    prompt = "The quick brown fox jumps over the lazy dog. " * 200
    results = []

    for i in range(3):
        print(f"\n📊 Request {i+1}/3...")
        start = time.perf_counter()
        ttft = None
        tokens = 0

        response = await client.chat.completions.create(
            model="qwen3.5-35b-mlx",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True}
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - start
            if hasattr(chunk, 'usage') and chunk.usage:
                tokens = chunk.usage.completion_tokens

        end = time.perf_counter()
        total = end - start
        gen_time = total - ttft if ttft else total
        tps = tokens / gen_time if gen_time > 0 else 0

        results.append({
            'ttft': ttft * 1000 if ttft else 0,
            'total': total,
            'tokens': tokens,
            'tps': tps
        })

        print(f"  TTFT: {ttft*1000:.1f}ms, TPS: {tps:.1f} tok/s")

    # Calculate averages
    avg_ttft = sum(r['ttft'] for r in results) / len(results)
    avg_tps = sum(r['tps'] for r in results) / len(results)

    print("\n" + "=" * 80)
    print("📈 Average Performance (3 requests)")
    print("=" * 80)
    print(f"  Avg TTFT: {avg_ttft:.1f}ms")
    print(f"  Avg Generation TPS: {avg_tps:.1f} tok/s")

    return avg_tps

async def main():
    # First run: profiled
    tps = await profile_server_api()

    # Then run multiple for average
    avg_tps = await run_multiple_requests()

    print("\n" + "=" * 80)
    print("🎯 FINAL RESULTS")
    print("=" * 80)
    print(f"  ThunderOMLX Server Generation TPS: {avg_tps:.1f} tok/s")
    print(f"  Original MLX Benchmark: 87.4 tok/s")
    print(f"  Difference: {avg_tps - 87.4:.1f} tok/s ({(avg_tps/87.4 - 1)*100:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())
