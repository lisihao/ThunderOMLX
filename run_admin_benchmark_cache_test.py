#!/usr/bin/env python3
"""
使用 oMLX Admin Panel 标准 benchmark 测试缓存影响
"""
import asyncio
import sys
import gc
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omlx.admin.benchmark import create_run, run_benchmark, BenchmarkRequest
from omlx.engine_pool import EnginePool
from omlx.scheduler import SchedulerConfig

async def run_test(cache_enabled: bool):
    """运行单次测试（缓存开启或关闭）"""
    print("="*80)
    if cache_enabled:
        print("📊 Admin Benchmark - 缓存开启")
    else:
        print("📊 Admin Benchmark - 缓存禁用")
    print("="*80)
    print()

    # 模型路径
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        return None

    # 创建 SchedulerConfig（控制缓存）
    if cache_enabled:
        scheduler_config = SchedulerConfig(
            max_num_seqs=16,
            # 默认配置，缓存开启
        )
        print("✅ 缓存配置: SSD 缓存开启")
    else:
        scheduler_config = SchedulerConfig(
            max_num_seqs=16,
            paged_ssd_cache_dir=None,  # 禁用 SSD 缓存
            hot_cache_max_size=0,       # 禁用 hot cache
        )
        print("✅ 缓存配置: 所有缓存禁用")

    print()

    # 创建 EnginePool
    print("📦 初始化 EnginePool...")
    engine_pool = EnginePool(
        max_model_memory=40 * 1024**3,  # 40GB
        scheduler_config=scheduler_config
    )

    # 发现模型
    model_dir = model_path.parent
    engine_pool.discover_models(str(model_dir))

    # 创建 BenchmarkRequest
    model_id = model_path.name
    request = BenchmarkRequest(
        model_id=model_id,
        prompt_lengths=[8192],  # 测试 8K context
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
    print("🚀 运行 benchmark...")
    print()

    # 运行 benchmark
    try:
        await run_benchmark(run, engine_pool)

        # 提取结果
        if run.results:
            result = run.results[0]
            print()
            print("="*80)
            print("✅ Benchmark 完成")
            print("="*80)
            print(f"⏱️  TTFT: {result.get('ttft_ms', 0):.1f}ms")
            print(f"⚡ Generation TPS: {result.get('gen_tps', 0):.1f} tok/s")
            print(f"📊 Processing TPS: {result.get('processing_tps', 0):.1f} tok/s")
            print(f"📊 Prefill time: {result.get('ttft_ms', 0)/1000:.3f}s")

            # 计算 PP TPS（使用 TTFT 和 prompt tokens）
            ttft_s = result.get('ttft_ms', 0) / 1000
            prompt_tokens = 8192
            if ttft_s > 0:
                pp_tps = prompt_tokens / ttft_s
                print(f"📊 PP TPS: {pp_tps:.1f} tok/s")

            return result
        else:
            print("⚠️  没有结果数据")
            return None

    except Exception as e:
        print(f"❌ Benchmark 失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 清理
        gc.collect()
        print()


async def main():
    print()
    print("🔬 oMLX Admin Benchmark - 缓存性能对比测试")
    print("="*80)
    print("测试工具: oMLX 社区标准 benchmark")
    print("测试配置: pp8192/tg128")
    print("="*80)
    print()

    # 测试 1: 缓存开启
    print("\n" + "🟢 "*40 + "\n")
    result_with_cache = await run_test(cache_enabled=True)

    # 等待一段时间
    await asyncio.sleep(3)
    gc.collect()

    # 测试 2: 缓存禁用
    print("\n" + "🔴 "*40 + "\n")
    result_no_cache = await run_test(cache_enabled=False)

    # 对比结果
    print()
    print("="*80)
    print("📊 对比结果（oMLX 官方 Benchmark）")
    print("="*80)
    print()

    if result_with_cache and result_no_cache:
        print(f"{'指标':<25} {'缓存开启':<20} {'缓存禁用':<20} {'差异'}")
        print("-"*80)

        # TTFT (Prefill time)
        ttft_cache = result_with_cache.get('ttft_ms', 0)
        ttft_no_cache = result_no_cache.get('ttft_ms', 0)
        ttft_diff = ((ttft_cache - ttft_no_cache) / ttft_no_cache * 100) if ttft_no_cache > 0 else 0
        print(f"{'TTFT (ms)':<25} {ttft_cache:<20.1f} {ttft_no_cache:<20.1f} {ttft_diff:+.1f}%")

        # PP TPS
        if ttft_cache > 0 and ttft_no_cache > 0:
            pp_tps_cache = 8192 / (ttft_cache / 1000)
            pp_tps_no_cache = 8192 / (ttft_no_cache / 1000)
            pp_diff = ((pp_tps_no_cache - pp_tps_cache) / pp_tps_cache * 100)
            print(f"{'PP TPS (tok/s)':<25} {pp_tps_cache:<20.1f} {pp_tps_no_cache:<20.1f} {pp_diff:+.1f}%")

        # TG TPS
        gen_tps_cache = result_with_cache.get('gen_tps', 0)
        gen_tps_no_cache = result_no_cache.get('gen_tps', 0)
        gen_diff = ((gen_tps_no_cache - gen_tps_cache) / gen_tps_cache * 100) if gen_tps_cache > 0 else 0
        print(f"{'Generation TPS (tok/s)':<25} {gen_tps_cache:<20.1f} {gen_tps_no_cache:<20.1f} {gen_diff:+.1f}%")

        # Processing TPS
        proc_tps_cache = result_with_cache.get('processing_tps', 0)
        proc_tps_no_cache = result_no_cache.get('processing_tps', 0)
        proc_diff = ((proc_tps_no_cache - proc_tps_cache) / proc_tps_cache * 100) if proc_tps_cache > 0 else 0
        print(f"{'Processing TPS (tok/s)':<25} {proc_tps_cache:<20.1f} {proc_tps_no_cache:<20.1f} {proc_diff:+.1f}%")

        print()
        print("🔍 结论:")
        if pp_diff > 15:
            print(f"   ✅ 缓存写入是性能瓶颈（禁用后 PP 提升 {pp_diff:.1f}%）")
        elif pp_diff > 5:
            print(f"   ⚠️  缓存写入有一定影响（禁用后 PP 提升 {pp_diff:.1f}%）")
        else:
            print(f"   ❌ 缓存写入影响较小（禁用后 PP 仅提升 {pp_diff:.1f}%）")
    else:
        print("❌ 无法对比结果（某个测试失败）")

if __name__ == "__main__":
    asyncio.run(main())
