#!/usr/bin/env python3
"""真实 Prefill Profiling 测试"""
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# 启用 profiling
os.environ['OMLX_ENABLE_PROFILING'] = 'true'
os.environ['OMLX_PROFILING_INTERVAL'] = '1'  # 每次 prefill 都打印


async def main():
    from omlx.engine.batched import BatchedEngine
    from omlx.scheduler import SchedulerConfig
    from omlx.profiling import get_global_profiler, print_profiling_stats
    from transformers import AutoTokenizer

    print("="*80)
    print("🔍 真实 Prefill Profiling 测试")
    print("="*80)

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    # 加载 tokenizer
    print("\n⏳ 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )

    # 生成 8K prompt
    print("⏳ 生成 8K prompt...")
    base_words = [
        "technology", "innovation", "development", "programming", "software",
        "architecture", "infrastructure", "implementation", "optimization",
    ]
    text = " ".join(base_words * 1000)
    tokens = tokenizer.encode(text)[:8192]
    prompt = tokenizer.decode(tokens)
    actual_tokens = len(tokens)
    print(f"✅ Prompt: {actual_tokens} tokens\n")

    # 初始化引擎（缓存禁用）
    print("⏳ 初始化引擎（缓存禁用）...")
    scheduler_config = SchedulerConfig(
        paged_ssd_cache_dir=None,
        hot_cache_max_size=0
    )
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True,
        scheduler_config=scheduler_config
    )

    await engine.start()
    print("✅ 引擎启动\n")

    try:
        # Test: Prefill with Profiling
        print("🚀 测试 Prefill（带 Profiling）...")
        print("-" * 80)

        start_time = time.perf_counter()
        first_token_time = None

        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=1,  # 只生成 1 个 token，测 prefill
            temperature=0.0
        ):
            if output.new_text:
                first_token_time = time.perf_counter()
                break

        if first_token_time:
            prefill_time = first_token_time - start_time
            pp_tps = actual_tokens / prefill_time

            print(f"\n✅ Prefill 完成")
            print(f"   时间: {prefill_time:.3f}s")
            print(f"   PP TPS: {pp_tps:.1f} tok/s")

        # 打印详细 profiling 统计
        print("\n" + "="*80)
        print("📊 详细 Profiling 统计")
        print("="*80)
        print_profiling_stats(top_n=30, min_percent=0.5)
        print("="*80)

        # 导出 JSON
        profiler = get_global_profiler()
        stats = profiler.get_stats()

        import json
        with open('/tmp/profiling_real_test.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print("\n💾 详细数据保存到: /tmp/profiling_real_test.json")

        # 分析瓶颈
        print("\n" + "="*80)
        print("🔍 瓶颈分析")
        print("="*80)

        bottlenecks = [
            (name, op_stats)
            for name, op_stats in stats['top_operations']
            if op_stats['percent'] > 5.0 and 'prefill.' in name
        ]

        if bottlenecks:
            print(f"\n发现 {len(bottlenecks)} 个主要瓶颈（占比 > 5%）:\n")
            for name, op_stats in bottlenecks:
                print(f"  ⚠️  {name}")
                print(f"      - 平均时间: {op_stats['avg_ms']:.2f} ms")
                print(f"      - 占比: {op_stats['percent']:.1f}%")
                if op_stats['count'] > 1:
                    print(f"      - 调用次数: {op_stats['count']}")
                print()

    finally:
        await engine.stop()
        print("✅ 引擎已关闭")


if __name__ == "__main__":
    asyncio.run(main())
