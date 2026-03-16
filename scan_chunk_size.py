#!/usr/bin/env python3
"""
Chunk Size 扫描测试
目标：找到最优的 chunk size（当前默认 512）
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import asyncio
from omlx.admin.benchmark import create_run, run_benchmark, BenchmarkRequest
from omlx.engine_pool import EnginePool
from omlx.scheduler import SchedulerConfig

async def test_chunk_size(engine_pool, chunk_size: int):
    """测试指定 chunk size 的性能"""
    print(f"\n{'=' * 70}")
    print(f"🧪 测试 Chunk Size: {chunk_size}")
    print(f"{'=' * 70}")

    # 临时修改配置
    from omlx import settings
    original_chunk_size = getattr(settings, 'CHUNKED_PREFILL_SIZE', 512)
    settings.CHUNKED_PREFILL_SIZE = chunk_size

    try:
        # 创建 BenchmarkRequest
        model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
        request = BenchmarkRequest(
            model_id=model_path.name,
            prompt_lengths=[8192],
            generation_length=128,
            batch_sizes=[],
            include_image=False
        )

        # 创建 BenchmarkRun
        run = create_run(request)

        # 运行 benchmark
        start = time.time()
        await run_benchmark(run, engine_pool)
        elapsed = time.time() - start

        # 收集结果
        result = None
        if run.results:
            result = run.results[0]

        return {
            'chunk_size': chunk_size,
            'elapsed': elapsed,
            'ttft_ms': result.get('ttft_ms', 0) if result else 0,
            'processing_tps': result.get('processing_tps', 0) if result else 0,
            'gen_tps': result.get('gen_tps', 0) if result else 0,
        }

    finally:
        # 恢复原配置
        settings.CHUNKED_PREFILL_SIZE = original_chunk_size


async def main():
    print("=" * 70)
    print("🎚️  Chunk Size 扫描测试")
    print("=" * 70)
    print()
    print("测试范围: 256, 384, 512, 768, 1024, 2048")
    print("目标: 找到 GPU 利用率和调度开销的最佳平衡点")
    print()

    # 模型路径
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        return

    # 创建 EnginePool（只创建一次）
    print("📦 初始化 EnginePool...")
    engine_pool = EnginePool(
        max_model_memory=40 * 1024**3,
        scheduler_config=SchedulerConfig(max_num_seqs=16)
    )

    # 发现模型
    print("🔍 发现模型...")
    engine_pool.discover_models(str(model_path.parent))
    print()

    chunk_sizes = [256, 384, 512, 768, 1024, 2048]
    results = []

    for chunk_size in chunk_sizes:
        result = await test_chunk_size(engine_pool, chunk_size)
        results.append(result)

        print(f"\n✅ Chunk {chunk_size}: "
              f"TTFT={result['ttft_ms']:.1f}ms, "
              f"TPS={result['processing_tps']:.1f} tok/s")

    # 保存结果
    output_file = '/tmp/chunk_size_scan_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 分析结果
    print("\n" + "=" * 70)
    print("📊 扫描结果汇总")
    print("=" * 70)
    print()
    print(f"{'Chunk Size':<12} {'TTFT (ms)':<12} {'TPS (tok/s)':<15} {'相对性能':<12}")
    print("-" * 70)

    baseline_tps = results[2]['processing_tps']  # 512 是基准

    for r in results:
        chunk_size = r['chunk_size']
        ttft = r['ttft_ms']
        tps = r['processing_tps']
        relative = ((tps / baseline_tps - 1) * 100) if baseline_tps > 0 else 0

        marker = ""
        if tps == max(r['processing_tps'] for r in results):
            marker = " ⭐ 最优"

        print(f"{chunk_size:<12} {ttft:<12.1f} {tps:<15.1f} {relative:>+6.1f}%{marker}")

    # 推荐
    best = max(results, key=lambda x: x['processing_tps'])
    print("\n" + "=" * 70)
    print("🎯 推荐配置")
    print("=" * 70)
    print(f"   最优 Chunk Size: {best['chunk_size']}")
    print(f"   Prefill TPS: {best['processing_tps']:.1f} tok/s")
    print(f"   TTFT: {best['ttft_ms']:.1f}ms")

    if best['chunk_size'] != 512:
        improvement = (best['processing_tps'] / baseline_tps - 1) * 100
        print(f"\n   💡 相比当前配置 (512)，提升 {improvement:+.1f}%")

    print(f"\n📁 详细结果保存到: {output_file}")


if __name__ == '__main__':
    asyncio.run(main())
