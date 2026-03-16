#!/usr/bin/env python3
"""
测试 cache 效果 - 对比 cache hit vs cache miss
"""
import asyncio
import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main():
    from omlx.engine.batched import BatchedEngine
    from transformers import AutoTokenizer

    print("=" * 70)
    print("📊 Cache 效果测试")
    print("=" * 70)
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        fix_mistral_regex=True  # 修复 Qwen tokenizer regex 问题
    )

    # 生成固定 prompt（8K tokens）
    base_words = ["technology", "innovation", "development"] * 1000
    text = " ".join(base_words)
    tokens = tokenizer.encode(text)[:8000]
    prompt = tokenizer.decode(tokens)
    actual_tokens = len(tokens)

    print(f"测试 prompt: {actual_tokens} tokens")
    print("")

    print("初始化引擎...")
    engine = BatchedEngine(model_name=str(model_path), trust_remote_code=True)
    await engine.start()
    print("✅ 引擎启动\n")

    try:
        results = []

        # 测试 3 次相同 prompt
        for i in range(3):
            print(f"\n{'─' * 70}")
            print(f"第 {i+1} 次测试（相同 prompt）")

            start = time.perf_counter()
            first_token = None

            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=1,
                temperature=0.0
            ):
                if output.new_text and first_token is None:
                    first_token = time.perf_counter()
                    break

            if first_token:
                prefill = first_token - start
                pp_tps = actual_tokens / prefill

                cache_status = "❄️ Cache MISS" if i == 0 else "🔥 Cache HIT"
                print(f"{cache_status}")
                print(f"   Prefill 时间: {prefill:.3f}s")
                print(f"   PP TPS: {pp_tps:.1f} tok/s")

                results.append({
                    'run': i + 1,
                    'prefill_time': prefill,
                    'pp_tps': pp_tps,
                    'cache_hit': i > 0
                })

            gc.collect()
            await asyncio.sleep(1)

        # 总结
        print("\n" + "=" * 70)
        print("📊 Cache 效果分析")
        print("=" * 70)

        if len(results) >= 2:
            miss = results[0]
            hits = results[1:]

            print(f"\n❄️  Cache MISS (第 1 次):")
            print(f"   Prefill: {miss['prefill_time']:.3f}s")
            print(f"   PP TPS: {miss['pp_tps']:.1f} tok/s")

            print(f"\n🔥 Cache HIT (第 2-3 次平均):")
            avg_hit_time = sum(r['prefill_time'] for r in hits) / len(hits)
            avg_hit_tps = sum(r['pp_tps'] for r in hits) / len(hits)
            print(f"   Prefill: {avg_hit_time:.3f}s")
            print(f"   PP TPS: {avg_hit_tps:.1f} tok/s")

            speedup = miss['prefill_time'] / avg_hit_time
            tps_gain = avg_hit_tps - miss['pp_tps']
            print(f"\n🚀 Cache 加速:")
            print(f"   时间加速: {speedup:.1f}x")
            print(f"   TPS 提升: +{tps_gain:.1f} tok/s ({(tps_gain/miss['pp_tps']*100):+.1f}%)")

        return True

    finally:
        print("\n关闭引擎...")
        await engine.stop()
        gc.collect()
        print("✅ 引擎已关闭")


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
