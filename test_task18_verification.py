#!/usr/bin/env python3
"""验证 Task #18 优化效果（使用正确的缓存预热流程）"""

import asyncio
import cProfile
import pstats
import time
import logging
from pathlib import Path
from mlx_lm import load

# 设置日志级别为 WARNING，减少噪音
logging.basicConfig(level=logging.WARNING)


async def warmup_phase():
    """Phase 1: Warm-up，建立缓存"""
    print("\n" + "=" * 80)
    print("Phase 1: Warm-up（建立缓存）")
    print("=" * 80)

    # 清空缓存目录
    cache_dir = Path.home() / ".cache" / "omlx" / "task18_verification"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    model, tokenizer = load(str(model_path))

    # 创建 EngineCore
    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig
    from omlx.request import SamplingParams

    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        paged_cache_block_size=256,
        disable_block_size_enlargement=True,
        max_cache_blocks=512,
        initial_cache_blocks=64,
        paged_ssd_cache_dir=str(cache_dir),
        model_name=str(model_path),
    )

    engine_config = EngineConfig(
        model_name=str(model_path),
        scheduler_config=scheduler_config,
    )

    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine.start()

    # Warm-up
    prompt = "请详细解释人工智能" * 2
    sampling_params = SamplingParams(max_tokens=1)
    await engine.generate(prompt=prompt, sampling_params=sampling_params)

    # 等待缓存写入
    await asyncio.sleep(1.0)

    print("✅ Warm-up 完成，缓存已写入")


async def profiling_phase():
    """Phase 2: Profiling，测量 FULL SKIP 性能"""
    print("\n" + "=" * 80)
    print("Phase 2: cProfile 性能测试（5 次 FULL SKIP 生成）")
    print("=" * 80)

    cache_dir = Path.home() / ".cache" / "omlx" / "task18_verification"

    # 加载模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    model, tokenizer = load(str(model_path))

    # 创建 EngineCore
    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig
    from omlx.request import SamplingParams

    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        paged_cache_block_size=256,
        disable_block_size_enlargement=True,
        max_cache_blocks=512,
        initial_cache_blocks=64,
        paged_ssd_cache_dir=str(cache_dir),
        model_name=str(model_path),
    )

    engine_config = EngineConfig(
        model_name=str(model_path),
        scheduler_config=scheduler_config,
    )

    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine.start()

    # 测试 prompt
    prompt = "请详细解释人工智能" * 2
    sampling_params = SamplingParams(max_tokens=1)

    # cProfile 性能测试
    profiler = cProfile.Profile()
    profiler.enable()

    # 运行 5 次 FULL SKIP 生成
    for i in range(5):
        await engine.generate(prompt=prompt, sampling_params=sampling_params)
        print(f"  完成第 {i+1} 次生成")

    profiler.disable()

    # 分析结果
    print("\n" + "=" * 80)
    print("cProfile 分析结果（按自身时间排序，Top 50）")
    print("=" * 80)

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('tottime')

    # 保存到文件
    with open('profiling/task18_verification_tottime.txt', 'w') as f:
        stats.stream = f
        stats.print_stats(50)

    # 输出到屏幕
    stats.stream = None
    stats.print_stats(50)

    print("\n" + "=" * 80)
    print("分析结果已保存到: profiling/task18_verification_tottime.txt")
    print("=" * 80)


async def main():
    """主流程：Warm-up → Profiling"""
    await warmup_phase()
    await profiling_phase()


if __name__ == "__main__":
    asyncio.run(main())
