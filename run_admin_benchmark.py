#!/usr/bin/env python3
"""
使用 Admin Panel 的 benchmark 方法测试 - 收集性能分析数据
"""
import asyncio
import sys
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omlx.admin.benchmark import create_run, run_benchmark, BenchmarkRequest
from omlx.engine_pool import EnginePool
from omlx.scheduler import SchedulerConfig

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
        prompt_lengths=[8192],
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

        # 从 results 列表中提取结果
        if run.results:
            result = run.results[0]  # 第一个结果
            print(f"⏱️  TTFT: {result.get('ttft_ms', 0):.1f}ms")
            print(f"⚡ Generation TPS: {result.get('gen_tps', 0):.1f} tok/s")
            print(f"📊 Processing TPS: {result.get('processing_tps', 0):.1f} tok/s")
        else:
            print("⚠️  没有结果数据")

        print()
        print("💡 性能对比（查看上面的 scheduler profiling 输出）：")
        print("   - 直接测试: 79.8 tok/s (12.52 ms/tok, batch_gen=12.46ms)")
        print("   - Benchmark: 查看上面的 ⏱️ Perf 输出")

    except Exception as e:
        print(f"❌ Benchmark 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
