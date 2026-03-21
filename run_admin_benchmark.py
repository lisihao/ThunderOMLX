#!/usr/bin/env python3
"""
使用 Admin Panel 的 benchmark 方法测试 - 收集性能分析数据
"""
import asyncio
import os
import sys
from pathlib import Path

# 启用 profiling
os.environ["OMLX_ENABLE_PROFILING"] = "true"

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omlx.admin.benchmark import create_run, run_benchmark, BenchmarkRequest
from omlx.engine_pool import EnginePool
from omlx.scheduler import SchedulerConfig
from omlx.profiling import get_global_profiler

async def main():
    print("="*80)
    print("🔍 Admin Benchmark - 性能分析测试")
    print("="*80)
    print()
    print("运行 pp8192/tg128 benchmark 并收集性能数据...")
    print("scheduler profiling 会每 50 tokens 打印一次性能统计")
    print()

    # 硬编码模型路径
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        return

    # 创建 EnginePool
    print("📦 初始化 EnginePool...")
    engine_pool = EnginePool(
        max_model_memory=40 * 1024**3,  # 40GB
        scheduler_config=SchedulerConfig(
            max_num_seqs=16
        )
    )

    # 发现模型
    print("🔍 发现模型...")
    model_dir = model_path.parent  # ~/models
    engine_pool.discover_models(str(model_dir))

    # 创建 BenchmarkRequest (使用模型名称，不是绝对路径)
    model_id = model_path.name  # "qwen3.5-35b-mlx"
    request = BenchmarkRequest(
        model_id=model_id,
        prompt_lengths=[1024, 4096, 8192, 16384, 32768],
        generation_length=128,
        batch_sizes=[],
        include_image=False
    )

    # 创建 BenchmarkRun
    run = create_run(request)

    print(f"📊 Benchmark ID: {run.bench_id}")
    print(f"📦 Model: {request.model_id}")
    print(f"📝 Config: pp{request.prompt_lengths[0]}/tg{request.generation_length}")
    print()
    print("🚀 开始运行 benchmark...")
    print("⏱️  Scheduler 会每 50 tokens 打印性能数据...")
    print()

    # 运行 benchmark
    try:
        await run_benchmark(run, engine_pool)

        # 显示结果
        print()
        print("="*80)
        print("✅ Benchmark 完成")
        print("="*80)

        # 显示所有结果
        if run.results:
            print(f"\n{'PP':>8} {'TG tok/s':>10} {'PP tok/s':>10} {'TTFT ms':>10}")
            print("-" * 42)
            for result in run.results:
                pp = result.get('pp', 0)
                gen_tps = result.get('gen_tps', 0)
                proc_tps = result.get('processing_tps', 0)
                ttft = result.get('ttft_ms', 0)
                print(f"{pp:>8} {gen_tps:>10.1f} {proc_tps:>10.1f} {ttft:>10.1f}")
        else:
            print("⚠️  没有结果数据")

        print()
        print("💡 性能对比（查看上面的 scheduler profiling 输出）：")
        print("   - 直接测试: 79.8 tok/s (12.52 ms/tok, batch_gen=12.46ms)")
        print("   - Benchmark: 查看上面的 ⏱️ Perf 输出")

        # 保存 profiling 报告
        print()
        print("="*80)
        print("📊 Profiling 报告")
        print("="*80)

        profiler = get_global_profiler()
        if profiler.enabled:
            # 打印统计
            profiler.print_stats(top_n=30, min_percent=0.5)

            # 保存 JSON
            json_path = "/tmp/mlx_lm_profiler_report.json"
            profiler.save_json(json_path)
            print(f"\n✅ Profiler 报告已保存: {json_path}")

            # 简单分析
            import json
            with open(json_path) as f:
                data = json.load(f)

            summary = data["summary"]
            cache_stats = data.get("cache_stats", {})

            print("\n" + "="*80)
            print("🔍 性能瓶颈分析")
            print("="*80)
            print(f"\n📊 整体指标:")
            print(f"  总请求数:        {summary.get('total_requests', 0)}")
            print(f"  Cache 命中率:    {summary.get('cache_hit_rate', 0):.2%}")

            if cache_stats:
                print(f"\n💾 Cache 统计:")
                for cache_type, stats in cache_stats.items():
                    print(f"  {cache_type:<20} {stats['hits']:>6} hits, {stats['misses']:>6} misses, {stats['hit_rate']:>6.2%}")

        else:
            print("⚠️  Profiling 未启用")

    except Exception as e:
        print(f"❌ Benchmark 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
