#!/usr/bin/env python3
"""
详细 Profiling Prefill 性能
目标：定位 1.5s "其他开销" 的具体来源
"""
import sys
import time
import cProfile
import pstats
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import asyncio
from omlx.admin.benchmark import create_run, run_benchmark, BenchmarkRequest
from omlx.engine_pool import EnginePool
from omlx.scheduler import SchedulerConfig

async def main():
    print("=" * 70)
    print("🔍 详细 Profiling Prefill 性能")
    print("=" * 70)
    print()
    print("目标：定位 1.5s '其他开销' 的具体来源")
    print()

    # 模型路径
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        return

    # 创建 EnginePool
    print("📦 初始化 EnginePool...")
    engine_pool = EnginePool(
        max_model_memory=40 * 1024**3,
        scheduler_config=SchedulerConfig(max_num_seqs=16)
    )

    # 发现模型
    print("🔍 发现模型...")
    engine_pool.discover_models(str(model_path.parent))

    # 创建 BenchmarkRequest
    request = BenchmarkRequest(
        model_id=model_path.name,
        prompt_lengths=[8192],
        generation_length=128,
        batch_sizes=[],
        include_image=False
    )

    # 创建 BenchmarkRun
    run = create_run(request)

    print(f"📊 Benchmark ID: {run.bench_id}")
    print(f"📦 Model: {request.model_id}")
    print()

    # 启动 profiling
    profiler = cProfile.Profile()

    print("⏱️  开始 profiling...")
    start = time.time()

    profiler.enable()
    await run_benchmark(run, engine_pool)
    profiler.disable()

    elapsed = time.time() - start
    print(f"\n✅ Profiling 完成，总耗时: {elapsed:.1f}s")

    # 生成详细报告
    print("\n" + "=" * 70)
    print("📊 生成分析报告...")
    print("=" * 70)

    # 1. 按 cumulative time 排序（总时间）
    s1 = StringIO()
    ps1 = pstats.Stats(profiler, stream=s1).sort_stats('cumulative')
    ps1.print_stats(50)

    with open('/tmp/prefill_profile_cumulative.txt', 'w') as f:
        f.write(s1.getvalue())
    print("✅ 保存到: /tmp/prefill_profile_cumulative.txt")

    # 2. 按 tottime 排序（自身时间，排除子函数）
    s2 = StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
    ps2.print_stats(50)

    with open('/tmp/prefill_profile_tottime.txt', 'w') as f:
        f.write(s2.getvalue())
    print("✅ 保存到: /tmp/prefill_profile_tottime.txt")

    # 3. 打印关键统计
    print("\n" + "=" * 70)
    print("🔑 关键发现（Top 15 by tottime）:")
    print("=" * 70)

    ps2.stream = sys.stdout
    ps2.print_stats(15)

    # 4. 显示性能结果
    if run.results:
        r = run.results[0]
        print("\n" + "=" * 70)
        print("📊 性能结果:")
        print("=" * 70)
        print(f"   TTFT: {r.get('ttft_ms', 0):.1f}ms")
        print(f"   Prefill TPS: {r.get('processing_tps', 0):.1f} tok/s")
        print(f"   Gen TPS: {r.get('gen_tps', 0):.1f} tok/s")
        print()

    print("\n💡 下一步:")
    print("   1. 查看 /tmp/prefill_profile_tottime.txt")
    print("   2. 识别除了 model.forward 和 cache I/O 之外的耗时函数")
    print("   3. 分析这些函数的优化空间")

if __name__ == '__main__':
    asyncio.run(main())
