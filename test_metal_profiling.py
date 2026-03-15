#!/usr/bin/env python3
"""Metal System Trace 性能分析脚本

使用方式:
1. 打开 Xcode Instruments
2. 选择 "Metal System Trace" 模板
3. 选择此脚本作为目标
4. 点击 Record 开始 profiling

分析重点:
- GPU Utilization（GPU 利用率，是否频繁空闲？）
- Kernel Duration & Count（内核时长和数量，是否大量短时内核？）
- Memory Load/Store（内存操作，是否频繁小数据传输？）
"""

import asyncio
import time
from pathlib import Path

from mlx_lm import load


async def metal_profiling():
    """专门用于 Metal System Trace 的性能分析"""

    print("=" * 80)
    print("Metal System Trace 性能分析")
    print("=" * 80)

    # 1. 加载模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    print(f"\n[1/4] 加载模型: {model_path}")
    model, tokenizer = load(str(model_path))
    print("✅ 模型加载完成")

    # 2. 创建 EngineCore
    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig
    from omlx.request import SamplingParams

    print("\n[2/4] 创建 EngineCore")
    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        paged_cache_block_size=256,
        disable_block_size_enlargement=True,
        max_cache_blocks=512,
        initial_cache_blocks=64,
        paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "metal_profile"),
        model_name=str(model_path),
    )

    engine_config = EngineConfig(
        model_name=str(model_path),
        scheduler_config=scheduler_config,
    )

    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine.start()
    print("✅ EngineCore 启动完成")

    # 3. Warm up（建立缓存）
    print("\n[3/4] Warm up（建立缓存）")
    prompt = "请详细解释人工智能" * 2
    sampling_params = SamplingParams(max_tokens=1)

    await engine.generate(prompt=prompt, sampling_params=sampling_params)
    print("✅ Warm up 完成")

    # 4. Profile 阶段（FULL SKIP）
    print("\n[4/4] Profile 阶段（FULL SKIP）")
    print("=" * 80)
    print("开始 profiling（运行 20 次，取详细数据）...")
    print("⚠️  请确保 Instruments 正在录制！")
    print("=" * 80)

    # 给用户时间确认 Instruments 已经开始录制
    print("\n倒计时 3 秒后开始...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("\n🔴 开始录制！\n")

    times = []
    for i in range(20):
        start = time.perf_counter()
        await engine.generate(prompt=prompt, sampling_params=sampling_params)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        # 每 5 次打印一次进度
        if (i + 1) % 5 == 0:
            avg = sum(times[-5:]) / 5
            print(f"  [{i+1}/20] 最近 5 次平均: {avg:.2f} ms")

    # 统计结果
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n" + "=" * 80)
    print("Profiling 完成！")
    print("=" * 80)
    print(f"平均时间: {avg_time:.2f} ms")
    print(f"最小时间: {min_time:.2f} ms")
    print(f"最大时间: {max_time:.2f} ms")
    print(f"标准差: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5:.2f} ms")

    print("\n⚠️  现在可以停止 Instruments 录制并分析数据！")
    print("\n分析重点:")
    print("  1. GPU Utilization - 查看 GPU 是否频繁空闲")
    print("  2. Kernel Count - 查看是否有大量短时内核启动")
    print("  3. Memory Bandwidth - 查看内存带宽利用率")
    print("  4. CPU-GPU Sync - 查看是否有频繁的同步点")


if __name__ == "__main__":
    asyncio.run(metal_profiling())
