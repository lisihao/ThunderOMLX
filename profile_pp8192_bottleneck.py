#!/usr/bin/env python3
"""
Profile ThunderOMLX at pp8192 to find performance bottleneck vs Native MLX.
"""
import cProfile
import pstats
import io
import time
import asyncio
from openai import AsyncOpenAI

async def profile_pp8192():
    print("="*80)
    print("🔍 ThunderOMLX pp8192 Performance Profiling")
    print("="*80)
    
    # Connect to server
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="not-needed"
    )
    
    # Generate 8192-token prompt (matching benchmark)
    filler = "The quick brown fox jumps over the lazy dog. " * 1000
    # Approximate 8192 tokens
    prompt = filler[:40000]  # Rough estimate
    
    print(f"\n📝 Prompt: ~8192 tokens")
    print("🎯 Target: 128 generation tokens")
    print("\n🚀 Starting profiled request...")
    
    # Profile the request
    profiler = cProfile.Profile()
    profiler.enable()
    
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
    profiler.disable()
    
    # Metrics
    total_time = end - start
    gen_time = total_time - ttft if ttft else total_time
    gen_tps = tokens / gen_time if gen_time > 0 else 0
    
    print(f"\n✅ Request completed")
    print(f"  TTFT: {ttft*1000:.1f}ms" if ttft else "  TTFT: N/A")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Tokens: {tokens}")
    print(f"  Generation TPS: {gen_tps:.1f} tok/s")
    
    # Compare
    print("\n" + "="*80)
    print("📊 Performance Comparison")
    print("="*80)
    print(f"  ThunderOMLX (this run): {gen_tps:.1f} tok/s")
    print(f"  ThunderOMLX (benchmark): 66.2 tok/s")
    print(f"  Native MLX: 80.7 tok/s")
    print(f"  Gap: {80.7 - gen_tps:.1f} tok/s ({(80.7 - gen_tps)/80.7*100:.1f}% slower)")
    
    # Profile analysis
    print("\n" + "="*80)
    print("📊 CLIENT-SIDE PROFILE (Top 30 by Cumulative Time)")
    print("="*80)
    print("⚠️  Note: This profiles the HTTP client, not the server process")
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Save
    with open("profile_pp8192_client.txt", "w") as f:
        f.write(f"ThunderOMLX pp8192 Client Profile\n")
        f.write(f"Generation TPS: {gen_tps:.1f} tok/s\n")
        f.write(f"vs Native MLX: {80.7 - gen_tps:.1f} tok/s slower\n\n")
        f.write(s.getvalue())
    
    print("\n💾 Client profile saved to: profile_pp8192_client.txt")
    print("\n⚠️  WARNING: Client profile shows HTTP overhead, not server bottleneck")
    print("   Need to profile server process directly for real bottleneck analysis")
    
    return gen_tps

async def main():
    await profile_pp8192()

if __name__ == "__main__":
    asyncio.run(main())
