#!/usr/bin/env python3
"""
快速测试禁用压缩后的性能
"""
import asyncio
import time
import json
from openai import AsyncOpenAI

async def test_pp8192_no_compression():
    print("="*80)
    print("🧪 ThunderOMLX pp8192/tg128 - 禁用压缩测试")
    print("="*80)

    # 读取 API key
    with open("/Users/lisihao/.omlx/settings.json") as f:
        api_key = json.load(f)["auth"]["api_key"]

    client = AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key=api_key
    )

    # 生成 8192 token prompt
    filler = "The quick brown fox jumps over the lazy dog. " * 1000
    prompt = filler[:40000]  # 约 8192 tokens

    print(f"\n📝 Prompt: ~8192 tokens")
    print("🎯 Generation: 128 tokens")
    print("\n🚀 Starting test...")

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

    # 计算指标
    total_time = end - start
    gen_time = total_time - ttft if ttft else total_time
    gen_tps = tokens / gen_time if gen_time > 0 else 0

    print(f"\n✅ Test completed")
    print(f"  TTFT: {ttft*1000:.1f}ms" if ttft else "  TTFT: N/A")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Tokens: {tokens}")
    print(f"  Generation TPS: {gen_tps:.1f} tok/s")

    # 对比
    print("\n" + "="*80)
    print("📊 Performance Comparison")
    print("="*80)
    print(f"  ThunderOMLX (禁用压缩): {gen_tps:.1f} tok/s")
    print(f"  ThunderOMLX (启用lz4):  66.2 tok/s")
    print(f"  Native MLX:            80.7 tok/s")
    print(f"  Community baseline:     71.3 tok/s")

    improvement = gen_tps - 66.2
    gap_to_native = 80.7 - gen_tps

    print(f"\n  改进: {'+' if improvement > 0 else ''}{improvement:.1f} tok/s ({improvement/66.2*100:+.1f}%)")
    print(f"  vs Native MLX: {'-' if gap_to_native > 0 else '+'}{abs(gap_to_native):.1f} tok/s")

    # 判断是否是瓶颈
    if improvement > 10:
        print(f"\n✅ 确认: lz4 压缩是主要瓶颈！")
    elif improvement > 5:
        print(f"\n⚠️  lz4 压缩有影响，但可能还有其他瓶颈")
    else:
        print(f"\n❌ lz4 压缩不是主要瓶颈，需要继续排查")

    return gen_tps

if __name__ == "__main__":
    asyncio.run(test_pp8192_no_compression())
